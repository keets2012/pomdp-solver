package pomdp.algorithms.pointbased;

import pomdp.algorithms.ValueIteration;
import pomdp.environments.FactoredPOMDP;
import pomdp.environments.FactoredPOMDP.BeliefType;
import pomdp.environments.POMDP;
import pomdp.utilities.AlphaVector;
import pomdp.utilities.BeliefState;
import pomdp.utilities.BeliefStateVector;
import pomdp.utilities.ExecutionProperties;
import pomdp.utilities.HeuristicPolicy;
import pomdp.utilities.JProf;
import pomdp.utilities.Logger;
import pomdp.utilities.Pair;
import pomdp.utilities.factored.FactoredBeliefState;
import pomdp.valuefunction.JigSawValueFunction;
import pomdp.valuefunction.LinearValueFunctionApproximation;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

public class PForwardSearchValueIteration extends ValueIteration {
    protected int m_iLimitedBeliefMDPState;
    protected int m_iLimitedBeliefObservation;
    protected LinearValueFunctionApproximation m_vDetermisticPOMDPValueFunction;
    protected BeliefState m_bsDeterministicPOMDPBeliefState;
    protected HeuristicPolicy m_hpPolicy;
    protected int m_iDepth;
    protected int m_iIteration, m_iInnerIteration;
    protected long m_lLatestADRCheck, m_cTimeInADR, m_lCPUTimeTotal, m_lIterationStartTime;
    protected Pair m_pComputedADRs;
    protected int[] m_aiStartStates;
    protected SortedMap<Double, Integer>[][] m_amNextStates;
    private HeuristicType m_htType;
    public static HeuristicType DEFAULT_HEURISTIC = HeuristicType.MDP;
    //初始化信念点集合B
    BeliefStateVector<BeliefState> vBeliefPoints = new BeliefStateVector<BeliefState>();

    protected Iterator<BeliefState> m_itCurrentIterationPoints;
    protected boolean m_bSingleValueFunction = true;
    protected boolean m_bRandomizedActions;
    protected double m_dFilteredADR = 0.0;
    protected JigSawValueFunction m_vfUpperBound;

    double gamma = 0.95;  //折扣因子
    protected int maxIterations = 500;  //规定最多迭代次数
    protected int iterations = 0; //当前迭代次数
    protected double minWidth = Double.MAX_VALUE; //本次迭代上下界的最小差值
    protected double threshold = 0.0; //裁剪信念点的增加阈值
    public static final double EPSILON = 0.5;
    protected double SumDelta = 0.0;
    protected double dDelta = 1.0;

    protected double maxADR = -Integer.MAX_VALUE; //当前最大ADR

    protected long maxExecutionTime = 20 * 60; //20min,秒

    public enum HeuristicType {
        MDP, ObservationAwareMDP, DeterministicTransitionsPOMDP, DeterministicObservationsPOMDP, DeterministicPOMDP, LimitedBeliefMDP, HeuristicPolicy
    }

    public PForwardSearchValueIteration(POMDP pomdp) {
        this(pomdp, DEFAULT_HEURISTIC);
    }

    public PForwardSearchValueIteration(POMDP pomdp, HeuristicPolicy hpPolicy) {
        //this( pomdp, HeuristicType.DeterministicObservationsPOMDP );
        //this( pomdp, HeuristicType.LimitedBeliefMDP );
        this(pomdp, HeuristicType.MDP);
        m_hpPolicy = hpPolicy;
    }

    public PForwardSearchValueIteration(POMDP pomdp, HeuristicType htType) {

        super(pomdp);

        m_htType = htType;
        m_iDepth = 0;
        m_iIteration = 0;
        m_iInnerIteration = 0;
        m_lLatestADRCheck = 0;
        m_cTimeInADR = 0;
        m_lCPUTimeTotal = 0;
        m_lIterationStartTime = 0;
        m_pComputedADRs = null;
        m_aiStartStates = null;
        m_vfMDP = null;
        m_bsDeterministicPOMDPBeliefState = null;
        m_vDetermisticPOMDPValueFunction = null;
        m_iLimitedBeliefObservation = -1;
        initHeuristic();
        //m_vLabeledBeliefs = new Vector<BeliefState>();
    }

    private void initHeuristic() {
        long lBefore = JProf.getCurrentThreadCpuTimeSafe(), lAfter = 0;
        if (m_htType == HeuristicType.MDP) {
            m_vfMDP = m_pPOMDP.getMDPValueFunction();
            m_vfMDP.valueIteration(1000, ExecutionProperties.getEpsilon());
        }
        //把b0加入B
        /* initialize the list of belief points with the initial belief state */
        vBeliefPoints.add(null, m_pPOMDP.getBeliefStateFactory().getInitialBeliefState());

        lAfter = JProf.getCurrentThreadCpuTimeSafe();
        Logger.getInstance().log("PFSVI", 0, "initHeurisitc", "Initialization time was " + (lAfter - lBefore) / 1000000);
    }

    public void NewIteration(POMDP pomdp) {
        //收敛时间
        long lCPUTimeBefore = 0, lCPUTimeAfter = 0, lCPUTimeTotal = 0;
        Pair<Double, Double> pComputedADRs = new Pair<Double, Double>(new Double(0.0), new Double(0.0));
        double width = 0.0;

        //初始化信念点集合B
        BeliefStateVector<BeliefState> vBeliefPoints = new BeliefStateVector<BeliefState>();

        //把b0加入B
        /* initialize the list of belief points with the initial belief state */
        vBeliefPoints.add(null, m_pPOMDP.getBeliefStateFactory().getInitialBeliefState());

        boolean isconvergence = false;
        int cBeliefPoints = 0;

        //开始迭代直至收敛
        while (!isconvergence) {
            //本次循环开始时间
            lCPUTimeBefore = JProf.getCurrentThreadCpuTimeSafe();

            cBeliefPoints = vBeliefPoints.size();
            //更新前的信念点集
            vBeliefPoints = expandPBVI(vBeliefPoints);  //点集的扩张，增加点扩张时条件的判断
            if (vBeliefPoints.size() == cBeliefPoints) {
                isconvergence = true;
            }

            //更新上界和下界，dDelta为下界更新前后最大的提升量
            dDelta = improveValueFunction(vBeliefPoints);

            iterations++;

            //ADR
            pComputedADRs = CalculateADRConvergence(m_pPOMDP, pComputedADRs);
            if (((Number) pComputedADRs.first()).doubleValue() > maxADR) {
                maxADR = ((Number) pComputedADRs.first()).doubleValue();
            }

            //本次循环结束时间
            lCPUTimeAfter = JProf.getCurrentThreadCpuTimeSafe();
            //本次循环使用时间
            lCPUTimeTotal += (lCPUTimeAfter - lCPUTimeBefore);
            if (iterations >= maxIterations || lCPUTimeTotal / 1000000000 >= maxExecutionTime) {
                isconvergence = true;
            }

            Logger.getInstance().logln("Iteration: " + iterations +
                    " |Vn|: = " + m_vValueFunction.size() +
                    " |B|: = " + vBeliefPoints.size() +
                    " Delta: = " + round(dDelta, 4) +
                    " CurrentTotalTime: " + lCPUTimeTotal / 1000000000 + "seconds");

            //在backup之后对信念点集进行裁剪
            if (!isconvergence) {
                for (int index = 0; index < vBeliefPoints.size(); index++) {
                    width = width(vBeliefPoints.get(index));
                    //计算本次迭代的最小上下界差值
                    if (width < minWidth) {
                        minWidth = width;
                    }
                    //裁剪去值函数更新后上下界差值小于阈值的信念点
                    if (width < m_dEpsilon / Math.pow(gamma, iterations)) {
                        vBeliefPoints.remove(index);
                        index--;
                    }
                }
                Logger.getInstance().logln(" Prune after backup |B|: " + vBeliefPoints.size());
                Logger.getInstance().logln("minWidth: " + minWidth);
                Logger.getInstance().logln("maxADR: " + maxADR);
                Logger.getInstance().logln();
            }
            threshold = minWidth;

            System.out.println("");
        }

        Logger.getInstance().logln("Finished " + " - time : " + lCPUTimeTotal / 1000000000 + "seconds" + " |BS| = " + vBeliefPoints.size() +
                " |V| = " + m_vValueFunction.size());
    }

    public void valueIteration(int cMaxSteps, double dEpsilon, double dTargetValue, int maxRunningTime, int numEvaluations) {

        //public void valueIteration( int cMaxSteps, double dEpsilon, double dTargetValue ){
        int iIteration = 0;
        boolean bDone = false;
        Pair pComputedADRs = new Pair();
        double dMaxDelta = 0.0;
        String sMsg = "";

        long lStartTime = System.currentTimeMillis(), lCurrentTime = 0;
        long lCPUTimeBefore = 0, lCPUTimeAfter = 0;
        Runtime rtRuntime = Runtime.getRuntime();

        long cDotProducts = AlphaVector.dotProductCount(), cVnChanges = 0, cStepsWithoutChanges = 0;
        m_cElapsedExecutionTime = 0;
        m_lCPUTimeTotal = 0;

        sMsg = "Starting " + getName() + " target ADR = " + round(dTargetValue, 3);
        Logger.getInstance().log("PFSVI", 0, "VI", sMsg);

        //initStartStateArray();
        m_pComputedADRs = new Pair();

        for (iIteration = 0; (iIteration < cMaxSteps) && !bDone; iIteration++) {
            lStartTime = System.currentTimeMillis();
            lCPUTimeBefore = JProf.getCurrentThreadCpuTimeSafe();
            AlphaVector.initCurrentDotProductCount();
            cVnChanges = m_vValueFunction.getChangesCount();
            m_iIteration = iIteration;
            m_iInnerIteration = 0;
            m_lLatestADRCheck = lCPUTimeBefore;
            m_cTimeInADR = 0;
            m_lIterationStartTime = lCPUTimeBefore;
            dMaxDelta = improveValueFunction();
            lCPUTimeAfter = JProf.getCurrentThreadCpuTimeSafe();
            lCurrentTime = System.currentTimeMillis();
            m_cElapsedExecutionTime += (lCurrentTime - lStartTime - m_cTimeInADR);
            m_cCPUExecutionTime += (lCPUTimeAfter - lCPUTimeBefore - m_cTimeInADR) / 1000000;
            m_lCPUTimeTotal += lCPUTimeAfter - lCPUTimeBefore - m_cTimeInADR;

            if (m_bTerminate)
                bDone = true;


            if (ExecutionProperties.getReportOperationTime()) {
                try {
                    sMsg = "G: - operations " + AlphaVector.getGComputationsCount() + " avg time " +
                            AlphaVector.getAvgGTime();
                    Logger.getInstance().log("PFSVI", 0, "VI", sMsg);

                    if (m_pPOMDP.isFactored() && ((FactoredPOMDP) m_pPOMDP).getBeliefType() == BeliefType.Factored) {
                        sMsg = "Tau: - operations " + FactoredBeliefState.getTauComputationCount() + " avg time " +
                                FactoredBeliefState.getAvgTauTime();
                        Logger.getInstance().log("PFSVI", 0, "VI", sMsg);

                    } else {
                        sMsg = "Tau: - operations " + m_pPOMDP.getBeliefStateFactory().getTauComputationCount() + " avg time " +
                                m_pPOMDP.getBeliefStateFactory().getAvgTauTime();
                        Logger.getInstance().log("PFSVI", 0, "VI", sMsg);
                    }
                    sMsg = "dot product - avg time = " + AlphaVector.getCurrentDotProductAvgTime();
                    Logger.getInstance().log("PFSVI", 0, "VI", sMsg);
                    sMsg = "avg belief state size " + m_pPOMDP.getBeliefStateFactory().getAvgBeliefStateSize();
                    Logger.getInstance().log("PFSVI", 0, "VI", sMsg);
                    sMsg = "avg alpha vector size " + m_vValueFunction.getAvgAlphaVectorSize();
                    Logger.getInstance().log("PFSVI", 0, "VI", sMsg);
                    AlphaVector.initCurrentDotProductCount();
                } catch (Exception e) {
                    Logger.getInstance().logln(e);
                }
            }
            if (((m_lCPUTimeTotal / 1000000000) >= 5) && (iIteration >= 10) && (iIteration % 5 == 0) &&
                    m_vValueFunction.getChangesCount() > cVnChanges &&
                    m_vValueFunction.size() > 5) {


                cStepsWithoutChanges = 0;
                bDone |= checkADRConvergence(m_pPOMDP, dTargetValue, pComputedADRs);

                sMsg = "PFSVI: Iteration " + iIteration +
                        " |Vn| = " + m_vValueFunction.size() +
                        " simulated ADR " + round(((Number) pComputedADRs.first()).doubleValue(), 3) +
                        " filtered ADR " + round(((Number) pComputedADRs.second()).doubleValue(), 3) +
                        " max delta " + round(dMaxDelta, 3) +
                        " depth " + m_iDepth +
                        " V(b0) " + round(m_vValueFunction.valueAt(m_pPOMDP.getBeliefStateFactory().getInitialBeliefState()), 2) +
                        " time " + (lCurrentTime - lStartTime) / 1000 +
                        " CPU time " + (lCPUTimeAfter - lCPUTimeBefore - m_cTimeInADR) / 1000000000 +
                        " CPU total " + m_lCPUTimeTotal / 1000000000 +
                        " #backups " + m_cBackups +
                        " V changes " + m_vValueFunction.getChangesCount() +
                        " #dot product " + AlphaVector.dotProductCount() +
                        " |BS| " + m_pPOMDP.getBeliefStateFactory().getBeliefStateCount() +
                        " memory: " +
                        " total " + rtRuntime.totalMemory() / 1000000 +
                        " free " + rtRuntime.freeMemory() / 1000000 +
                        " max " + rtRuntime.maxMemory() / 1000000 +
                        "";
                Logger.getInstance().log("PFSVI", 0, "VI", sMsg);
            } else {
                if (cVnChanges == m_vValueFunction.getChangesCount()) {
                    cStepsWithoutChanges++;
                    //if( cStepsWithoutChanges == 10 ){
                    //	bDone = true;
                    //}
                }
                sMsg = "PFSVI: Iteration " + iIteration +
                        " |Vn| = " + m_vValueFunction.size() +
                        " time " + (lCurrentTime - lStartTime) / 1000 +
                        " V changes " + m_vValueFunction.getChangesCount() +
                        " max delta " + round(dMaxDelta, 3) +
                        " depth " + m_iDepth +
                        " V(b0) " + round(m_vValueFunction.valueAt(m_pPOMDP.getBeliefStateFactory().getInitialBeliefState()), 2) +
                        " CPU time " + (lCPUTimeAfter - lCPUTimeBefore) / 1000000000 +
                        " CPU total " + m_lCPUTimeTotal / 1000000000 +
                        " #backups " + m_cBackups +
                        " |BS| " + m_pPOMDP.getBeliefStateFactory().getBeliefStateCount() +
                        " memory: " +
                        " total " + rtRuntime.totalMemory() / 1000000 +
                        " free " + rtRuntime.freeMemory() / 1000000 +
                        " max " + rtRuntime.maxMemory() / 1000000 +
                        "";
                Logger.getInstance().log("PFSVI", 0, "VI", sMsg);


            }

        }
        m_bConverged = true;

        m_cDotProducts = AlphaVector.dotProductCount() - cDotProducts;
        m_cElapsedExecutionTime /= 1000;
        m_cCPUExecutionTime /= 1000;

        sMsg = "Finished " + getName() + " - time : " + m_cElapsedExecutionTime + /*" |BS| = " + vBeliefPoints.size() +*/
                " |V| = " + m_vValueFunction.size() +
                " backups = " + m_cBackups +
                " GComputations = " + AlphaVector.getGComputationsCount() +
                " Dot products = " + m_cDotProducts;
        Logger.getInstance().log("PFSVI", 0, "VI", sMsg);
    }

    protected double forwardSearch(int iState, BeliefState bsCurrent, BeliefState initBs, int iDepth) {
        double dDelta = 0.0, dNextDelta = 0.0;
        int iNextState = 0, iHeuristicAction = 0, iPOMDPAction = 0, iObservation = 0;
        BeliefState bsNext = null;
        AlphaVector avBackup = null, avMax = null;
        double dPreviousValue = 0.0, dNewValue = 0.0;

        if (m_bTerminate)
            return 0.0;

        if ((m_pPOMDP.terminalStatesDefined() && isTerminalState(iState))
                || (iDepth >= 100)) {
            m_iDepth = iDepth;
            Logger.getInstance().logln("Ended at depth " + iDepth + ". isTerminalState(" + iState + ")=" + isTerminalState(iState));
        } else {
//            iHeuristicAction = getAction(iState, bsCurrent);
            iHeuristicAction = getAction(iState, initBs);

            iNextState = selectNextState(iState, iHeuristicAction);
            iObservation = getObservation(iState, iHeuristicAction, iNextState);
            bsNext = bsCurrent.nextBeliefState(iHeuristicAction, iObservation);
            if (iObservation > 1) {
                vBeliefPoints.add(bsNext);
            }

            if (bsNext == null || bsNext.equals(bsCurrent)) {
                m_iDepth = iDepth;
            } else {

                double d = bsNext.valueAt(iNextState);
                if (bsNext.valueAt(iNextState) == 0.0) {
                    bsNext = bsCurrent.nextBeliefState(iHeuristicAction, iObservation);
                }
                dNextDelta = forwardSearch(iNextState, bsNext, initBs, iDepth + 1);
            }
        }

        if (true) {
            BeliefState bsDeterministic = getDeterministicBeliefState(iState);
            avBackup = backup(bsDeterministic, iHeuristicAction);
            dPreviousValue = m_vValueFunction.valueAt(bsDeterministic);
            dNewValue = avBackup.dotProduct(bsDeterministic);
            dDelta = dNewValue - dPreviousValue;

            if (dDelta > ExecutionProperties.getEpsilon()) {
                m_vValueFunction.addPrunePointwiseDominated(avBackup);
            }
        }
        avBackup = backup(bsCurrent);

        dPreviousValue = m_vValueFunction.valueAt(bsCurrent);
        dNewValue = avBackup.dotProduct(bsCurrent);
        dDelta = dNewValue - dPreviousValue;
        avMax = m_vValueFunction.getMaxAlpha(bsCurrent);

        if (dDelta > 0.0) {
            m_vValueFunction.addPrunePointwiseDominated(avBackup);
        } else {
            avBackup.release();
        }

        return Math.max(dDelta, dNextDelta);
    }

    protected BeliefStateVector<BeliefState> expandPBVI(BeliefStateVector<BeliefState> vBeliefPoints) {
        //扩充后的B，原先的B中内容已经在这里
        BeliefStateVector<BeliefState> vExpanded = new BeliefStateVector<BeliefState>(vBeliefPoints);
        Iterator it = vBeliefPoints.iterator();
        //临时变量，存放当前用来扩充的b
        BeliefState bsCurrent = null;
        //临时变量，存放得到的最远b
        BeliefState bsNext = null;

        //设置不需要缓存b
        boolean bPrevious = m_pPOMDP.getBeliefStateFactory().cacheBeliefStates(false);
        //每次扩充100个b
        int beliefsize = vBeliefPoints.size() + 100 < vBeliefPoints.size() * 2 ? vBeliefPoints.size() + 100 : vBeliefPoints.size() * 2;
        while (vExpanded.size() < beliefsize) {
            //是从扩充后B中随机取个b，计算它的最远后继！！和标准PBVI中expand不同
            //一个原因：保证能够扩充100个b
            bsCurrent = vExpanded.elementAt(m_rndGenerator.nextInt(vExpanded.size()));

            //计算最远的后继
            bsNext = m_pPOMDP.getBeliefStateFactory().computeLimitedFarthestSuccessor(vBeliefPoints, bsCurrent, iterations, m_vfUpperBound, m_vValueFunction, m_dEpsilon, gamma, threshold);
            if ((bsNext != null) && (!vExpanded.contains(bsNext)))
                vExpanded.add(bsCurrent, bsNext);
        }
        //设置回原来的值，是否要缓存b
        m_pPOMDP.getBeliefStateFactory().cacheBeliefStates(bPrevious);

        return vExpanded;
    }

    private boolean isTerminalState(int iState) {
        return m_pPOMDP.isTerminalState(iState);
    }

    private BeliefState getDeterministicBeliefState(int iState) {
        return m_pPOMDP.getBeliefStateFactory().getDeterministicBeliefState(iState);
    }

    private int getAction(int iState, BeliefState bs) {
        if (m_htType == HeuristicType.MDP) {
            if (m_rndGenerator.nextDouble() < 0.9)
                return m_vfMDP.getPriAction(iState, bs);
            return m_rndGenerator.nextInt(m_cActions);
        } else if (m_htType == HeuristicType.HeuristicPolicy) {
            return m_hpPolicy.getBestAction(iState, bs);
        }
        return -1;
    }

    private int getObservation(int iStartState, int iAction, int iEndState) {
        if (m_htType == HeuristicType.MDP) {
            return m_pPOMDP.observe(iAction, iEndState);
        } else if (m_htType == HeuristicType.HeuristicPolicy) {
            int iObservation = m_hpPolicy.getObservation(iStartState, iAction, iEndState);
            if (iObservation == -1)
                return m_pPOMDP.observe(iAction, iEndState);
            return iObservation;
        }
        return -1;
    }

    private int selectNextState(int iState, int iAction) {
        if (m_htType == HeuristicType.MDP) {
            return m_pPOMDP.execute(iAction, iState);
        } else if (m_htType == HeuristicType.HeuristicPolicy) {
            int iNextState = m_hpPolicy.getNextState(iState, iAction);
            if (iNextState == -1)
                return m_pPOMDP.execute(iAction, iState);
        }
        return -1;
    }

    private void removeNextState(int iState, int iAction, int iNextState) {
        m_amNextStates[iState][iAction].remove(m_amNextStates[iState][iAction].lastKey());
    }

    protected int getNextState(int iAction, int iState) {
        int iNextState = -1;
        double dTr = 0.0, dValue = 0.0;

        if (m_amNextStates[iState][iAction] == null)
            m_amNextStates[iState][iAction] = new TreeMap<Double, Integer>();
        if (m_amNextStates[iState][iAction].isEmpty()) {
            Iterator itNonZero = m_pPOMDP.getNonZeroTransitions(iState, iAction);
            Entry e = null;
            String sDescription = "";
            while (itNonZero.hasNext()) {
                e = (Entry) itNonZero.next();
                iNextState = ((Number) e.getKey()).intValue();
                dTr = ((Number) e.getValue()).doubleValue();
                dValue = m_vfMDP.getValue(iNextState);
                sDescription += "V(" + iNextState + ") = " + round(dValue, 3) + ", ";
                m_amNextStates[iState][iAction].put(dValue, iNextState);
            }
        }
        iNextState = m_amNextStates[iState][iAction].get(m_amNextStates[iState][iAction].lastKey());
        return iNextState;
    }

    protected void initStartStateArray() {
        int cStates = m_pPOMDP.getStartStateCount(), iState = 0;
        Iterator<Entry<Integer, Double>> itStartStates = m_pPOMDP.getStartStates();
        Entry<Integer, Double> e = null;
        m_aiStartStates = new int[cStates];
        for (iState = 0; iState < cStates; iState++) {
            e = itStartStates.next();
            m_aiStartStates[iState] = e.getKey();
        }
        if (m_amNextStates == null) {
            m_amNextStates = new SortedMap[m_cStates][m_cActions];
        }
    }

    protected int chooseStartState() {
        int cStates = m_pPOMDP.getStartStateCount(), iState = 0, iMaxValueState = -1;
        double dValue = 0.0, dMaxValue = MIN_INF;
        for (iState = 0; iState < cStates; iState++) {
            if (m_aiStartStates[iState] != -1) {
                dValue = m_vfMDP.getValue(iState);
                if (dValue > dMaxValue) {
                    dMaxValue = dValue;
                    iMaxValueState = iState;
                }
            }
        }
        if (iMaxValueState == -1) {
            initStartStateArray();
            return chooseStartState();
        }
        iState = m_aiStartStates[iMaxValueState];
        m_aiStartStates[iMaxValueState] = -1;
        return iState;
    }


    protected double improveValueFunction(BeliefStateVector vBeliefPoints) {
        LinearValueFunctionApproximation vNextValueFunction = new LinearValueFunctionApproximation(m_dEpsilon, true);
        BeliefState bsCurrent = null, bsMax = null;
        AlphaVector avBackup = null, avNext = null, avCurrentMax = null;
        double dMaxDelta = 1.0, dDelta = 0.0, dBackupValue = 0.0, dValue = 0.0;
        double dMaxOldValue = 0.0, dMaxNewValue = 0.0;
        int iBeliefState = 0;

        double maxUpperDecline = 0.0, upperDecline = 0.0; //上界下降值
        BeliefState upperState = null; //上界下降值最大的信念点

        boolean bPrevious = m_pPOMDP.getBeliefStateFactory().cacheBeliefStates(false);

        if (m_itCurrentIterationPoints == null)
            m_itCurrentIterationPoints = vBeliefPoints.getTreeDownUpIterator();
        dMaxDelta = 0.0;

        //迭代所有的b
        while (m_itCurrentIterationPoints.hasNext()) {
            //当前的b
            bsCurrent = (BeliefState) m_itCurrentIterationPoints.next();
            //当前b对应的最大α
            avCurrentMax = m_vValueFunction.getMaxAlpha(bsCurrent);
            //backup操作后的α
            avBackup = backup(bsCurrent);

            //计算backup前后，该b点value之差
            dBackupValue = avBackup.dotProduct(bsCurrent);
            dValue = avCurrentMax.dotProduct(bsCurrent);
            dDelta = dBackupValue - dValue;


            if (dDelta > dMaxDelta) {
                dMaxDelta = dDelta;
                bsMax = bsCurrent;
                dMaxOldValue = dValue;
                dMaxNewValue = dBackupValue;
            }

            avNext = avBackup;

            //如果有提升，才会增加新的α
            if (dDelta >= 0) {
                m_vValueFunction.addPrunePointwiseDominated(avBackup);
                SumDelta += dDelta;
            }

            //更新上界
            upperDecline = m_vfUpperBound.updateValue(bsCurrent);
            //获得本次更新上界下降的最大值
            if (upperDecline > maxUpperDecline) {
                maxUpperDecline = upperDecline;
                upperState = bsCurrent;
            }

            iBeliefState++;
        }

        if (m_bSingleValueFunction) {
            Iterator it = vNextValueFunction.iterator();
            while (it.hasNext()) {
                avNext = (AlphaVector) it.next();
                m_vValueFunction.addPrunePointwiseDominated(avNext);
            }
        }

        if (!m_itCurrentIterationPoints.hasNext())
            m_itCurrentIterationPoints = null;

        Logger.getInstance().logln("Max lowBounddelta over " + bsMax +
                " from " + round(dMaxOldValue, 3) +
                " to " + round(dMaxNewValue, 3));

        Logger.getInstance().logln("Max upperBounddelta over " + upperState +
                " is " + round(maxUpperDecline, 3));

        m_pPOMDP.getBeliefStateFactory().cacheBeliefStates(bPrevious);

        return dMaxDelta;
    }

    protected double improveValueFunction() {
        int iInitialState = -1;
        do {
            iInitialState = m_pPOMDP.chooseStartState();
        } while (m_pPOMDP.isTerminalState(iInitialState));
        BeliefState bsInitial = m_pPOMDP.getBeliefStateFactory().getInitialBeliefState();
        Logger.getInstance().logln("Starting at state " + m_pPOMDP.getStateName(iInitialState));
        m_iDepth = 0;
        Logger.getInstance().logln("Begin improve");
        double dDelta = forwardSearch(iInitialState, bsInitial, bsInitial, 0);
        Logger.getInstance().logln("End improve, |V| = " +
                m_vValueFunction.size() + ", delta = " + dDelta);
        return dDelta;
    }

}
