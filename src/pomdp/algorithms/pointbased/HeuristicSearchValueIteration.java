package pomdp.algorithms.pointbased;

import java.util.List;
import java.util.Map;
import java.util.Vector;

import pomdp.algorithms.ValueIteration;
import pomdp.environments.POMDP;
import pomdp.integral.Beta;
import pomdp.utilities.AlphaVector;
import pomdp.utilities.BeliefState;
import pomdp.utilities.ExecutionProperties;
import pomdp.utilities.JProf;
import pomdp.utilities.Logger;
import pomdp.utilities.Pair;
import pomdp.utilities.datastructures.LinkedList;
import pomdp.utilities.distribution.DistributionCalculator;
import pomdp.valuefunction.JigSawValueFunction;
import pomdp.valuefunction.LinearValueFunctionApproximation;
import pomdp.valuefunction.MDPValueFunction;

public class HeuristicSearchValueIteration extends ValueIteration {
    protected JigSawValueFunction m_vfUpperBound;
    protected int m_cApplyHComputations;
    protected int m_cNewPointComputations;
    protected int m_cVisitedBeliefStates;
    protected double m_dMaxWidthForIteration;
    private static double m_dExplorationFactor;

    private String algorithmName;
    private DistributionCalculator calculator;

    public HeuristicSearchValueIteration(POMDP pomdp, double dExplorationFactor, boolean bUseFIB) {
        super(pomdp);


        if (!m_vfMDP.persistQValues()) {
            m_vfMDP = new MDPValueFunction(pomdp, 0.0);
            m_vfMDP.persistQValues(true);
            m_vfMDP.valueIteration(100, m_dEpsilon);
        }
        m_vfUpperBound = new JigSawValueFunction(pomdp, m_vfMDP, bUseFIB);
        m_cNewPointComputations = 0;
        m_cApplyHComputations = 0;
        m_cVisitedBeliefStates = 0;
        m_dMaxWidthForIteration = 0.0;
        m_dExplorationFactor = dExplorationFactor;
    }

    public HeuristicSearchValueIteration(POMDP pomdp, boolean bUseFIB) {
        this(pomdp, 0.0, bUseFIB);
    }

    public void setAlgorithmName(String algorithmName) {
        this.algorithmName = algorithmName;
    }

    public String getAlgorithmName() {
        return this.algorithmName;
    }

    protected void applyH(BeliefState bs) {
        long lTimeBefore = 0, lTimeAfter = 0;

        if (ExecutionProperties.getReportOperationTime())
            lTimeBefore = JProf.getCurrentThreadCpuTimeSafe();

        m_vfUpperBound.updateValue(bs);

        if (ExecutionProperties.getReportOperationTime()) {
            lTimeAfter = JProf.getCurrentThreadCpuTimeSafe();

            m_cTimeInHV += (lTimeAfter - lTimeBefore) / 1000000;
        }
    }

    public int getAction(BeliefState bsCurrent) {
        AlphaVector avMaxAlpha = m_vValueFunction.getMaxAlpha(bsCurrent);
        return avMaxAlpha.getAction();
    }

    protected String toString(double[][] adArray) {
        int i = 0, j = 0;
        String sRes = "";
        for (i = 0; i < adArray.length; i++) {
            for (j = 0; j < adArray[i].length; j++) {
                sRes += adArray[i][j] + " ";
            }
            sRes += "\n";
        }
        return sRes;
    }

    protected double excess(BeliefState bsCurrent, double dEpsilon, double dDiscount) {
        return width(bsCurrent) - (dEpsilon / dDiscount);
    }

    protected double width(BeliefState bsCurrent) {
        double dUpperValue = 0.0, dLowerValue = 0.0, dWidth = 0.0;
        dUpperValue = m_vfUpperBound.valueAt(bsCurrent);
        dLowerValue = valueAt(bsCurrent);
        dWidth = dUpperValue - dLowerValue;

        return dWidth;
    }

    public String getName() {
        if (algorithmName == null) {
            return "HSVI";
        } else {
            return "HSVI" + "_" + algorithmName;
        }
    }

    public void valueIteration(int cIterations, double dEpsilon, double dTargetValue, int maxRunningTime, int numEvaluations) {
        // ��ʼ�����ֱ���
        // ��ʼ�����
        BeliefState bsInitial = m_pPOMDP.getBeliefStateFactory().getInitialBeliefState();
        // ��ʼ�������
        double dInitialWidth = width(bsInitial);

        // ���������� / ���explore���
        int iIteration = 0, iMaxDepth = 0;

        // ʱ�����
        long lStartTime = System.currentTimeMillis(), lCurrentTime = 0;

        // ���л���
        Runtime rtRuntime = Runtime.getRuntime();

        // ����ѭ����ֹ�ı���
        boolean bDone = false;

        // ��¼�����м����ADR
        Pair<Double, Double> pComputedADRs = new Pair<Double, Double>();

        //�۲⵽�������
        Vector<BeliefState> vObservedBeliefStates = new Vector<BeliefState>();
        int cUpperBoundPoints = 0, cNoChange = 0;
        String sMsg = "";

        m_cElapsedExecutionTime = 0;
        m_cCPUExecutionTime = 0;

        //����ʱ��
        long lCPUTimeBefore = 0, lCPUTimeAfter = 0, lCPUTimeTotal = 0;

        int cValueFunctionChanges = 0;

        //���ִ��ʱ�䣬Ĭ������Ϊ10����
        long maxExecutionTime = m_maxExecutionTime;//1000*60*10;

        Logger.getInstance().logln("Begin " + getName() + ", Initial width = " + dInitialWidth);

        // ѭ������
        // ����������ѭ������ָ������ || �ﵽ��ֹ����������ʱ�䣩
        for (iIteration = 0; (iIteration < cIterations) && !bDone && !m_bTerminate; iIteration++) {
            // ��ȡ������ʼ��ʱ���
            lStartTime = System.currentTimeMillis();
            lCPUTimeBefore = JProf.getCurrentThreadCpuTimeSafe();

            // ��¼�����������
            m_dMaxWidthForIteration = 0.0;

            // explore���������;����explore��������
            iMaxDepth = explore(bsInitial, dEpsilon, 0, 1.0, vObservedBeliefStates);

            // ���Ͻ纯���ĵ�ĸ�������1000 && ���ε���֮���Ͻ纯���ĵ�ĸ�����������10%; ���Ͻ纯�����вü���
            if ((m_vfUpperBound.getUpperBoundPointCount() > 1000) && (m_vfUpperBound.getUpperBoundPointCount() > cUpperBoundPoints * 1.1)) {
                // �ü����̣�ȥ���Ͻ纯���У�����ֵ�ͳ�ʼֵ�Ĳ�С��epsilon�ĵ�
                m_vfUpperBound.pruneUpperBound();
                cUpperBoundPoints = m_vfUpperBound.getUpperBoundPointCount();
            }

            // �ۼ�explore���
            m_cVisitedBeliefStates += iMaxDepth;

            // ���¼���b0���width
            dInitialWidth = width(bsInitial);

            //����ʱ��
            lCurrentTime = System.currentTimeMillis();
            lCPUTimeAfter = JProf.getCurrentThreadCpuTimeSafe();
            m_cElapsedExecutionTime += (lCurrentTime - lStartTime);
            m_cCPUExecutionTime += (lCPUTimeAfter - lCPUTimeBefore) / 1000000;
            lCPUTimeTotal += lCPUTimeAfter - lCPUTimeBefore;

            //���maxExecutionTime > 0 ������ʱ�䳬��maxExecutionTime;��ֹ����
            if (maxExecutionTime > 0) {
                bDone = (m_cElapsedExecutionTime > maxExecutionTime);
            }

            // ÿ5�����½纯�������仯ʱ������ADR
            if ((iIteration >= 5) && ((lCPUTimeTotal / 1000000000) >= 0) && (iIteration % 5 == 0) && m_vValueFunction.getChangesCount() > cValueFunctionChanges) {
                //ԭʼ�㷨;����ADR��ֵ�Ƿ���ڸ���ֵ�����������������
                //bDone = checkADRConvergence( m_pPOMDP, dTargetValue, pComputedADRs );
                //�������������Ϊ��ֹ����������
                checkADRConvergence(m_pPOMDP, dTargetValue, pComputedADRs);

                // ��¼�����½纯���ı仯ֵ
                cValueFunctionChanges = m_vValueFunction.getChangesCount();

                //��ʽ���� ��������
                rtRuntime.gc();

                //�����Ϣ
                sMsg = getName() + ": Iteration " + iIteration +
                        " initial width " + round(dInitialWidth, 3) +
                        " V(b) " + round(m_vValueFunction.valueAt(m_pPOMDP.getBeliefStateFactory().getInitialBeliefState()), 4) +
                        " V^(b) " + round(m_vfUpperBound.valueAt(m_pPOMDP.getBeliefStateFactory().getInitialBeliefState()), 4) +
                        //" max width bs = " + bsMaxWidth +
                        //" max width " + round( width( bsMaxWidth ), 3 ) +
                        " max depth " + iMaxDepth +
                        " max width " + round(m_dMaxWidthForIteration, 3) +
                        //" simulated ADR " + ((Number) pComputedADRs.first()).doubleValue() +
                        //" filtered ADR " + round( ((Number) pComputedADRs.second()).doubleValue(), 3 ) +
                        " Time " + (lCurrentTime - lStartTime) / 1000 +
                        " CPU time " + (lCPUTimeAfter - lCPUTimeBefore) / 1000000000 +
                        " CPU total " + lCPUTimeTotal / 1000000000 +
                        " |V| " + m_vValueFunction.size() +
                        " |V^| " + m_vfUpperBound.getUpperBoundPointCount() +
                        " V changes " + m_vValueFunction.getChangesCount() +
                        " #ObservedBS = " + vObservedBeliefStates.size() +
                        " #BS " + m_pPOMDP.getBeliefStateFactory().getBeliefUpdatesCount() +
                        " #backups " + m_cBackups +
                        " #V^(b) = " + m_cNewPointComputations +
                        " max depth " + iMaxDepth +
                        " free memory " + rtRuntime.freeMemory() / 1000000 +
                        " total memory " + rtRuntime.totalMemory() / 1000000 +
                        " max memory " + rtRuntime.maxMemory() / 1000000;
            } else {
                //�����Ϣ
                sMsg = getName() + ": Iteration " + iIteration +
                        " initial width " + round(dInitialWidth, 3) +
                        " V(b) " + round(m_vValueFunction.valueAt(m_pPOMDP.getBeliefStateFactory().getInitialBeliefState()), 4) +
                        " V^(b) " + round(m_vfUpperBound.valueAt(m_pPOMDP.getBeliefStateFactory().getInitialBeliefState()), 4) +
                        " max depth " + iMaxDepth +
                        " max width " + round(m_dMaxWidthForIteration, 3) +
                        " Time " + (lCurrentTime - lStartTime) / 1000 +
                        " CPU time " + (lCPUTimeAfter - lCPUTimeBefore) / 1000000000 +
                        " CPU total " + lCPUTimeTotal / 1000000000 +
                        " |V| " + m_vValueFunction.size() +
                        " |V^| " + m_vfUpperBound.getUpperBoundPointCount() +
                        " #ObservedBS = " + vObservedBeliefStates.size() +
                        " #BS " + m_pPOMDP.getBeliefStateFactory().getBeliefUpdatesCount() +
                        " #backups " + m_cBackups +
                        " #V^(b) = " + m_cNewPointComputations +
                        " #HV(B) = " + m_cApplyHComputations +
                        " free memory " + rtRuntime.freeMemory() / 1000000 +
                        " total memory " + rtRuntime.totalMemory() / 1000000 +
                        " max memory " + rtRuntime.maxMemory() / 1000000 +
                        "\t";
            }
            Logger.getInstance().log(getName(), 0, "VI", sMsg);
            Logger.getInstance().logln();

            //��¼�½纯���������仯�Ĵ���
            if (m_vValueFunction.getChangesCount() == cValueFunctionChanges) {
                cNoChange++;
            }
            //�����仯������
            else
                cNoChange = 0;
        }

        m_cElapsedExecutionTime /= 1000;
        m_cCPUExecutionTime /= 1000;

        //�����ɵ���Ϣ
        sMsg = "Finished " + getName() + " - time : " + m_cElapsedExecutionTime +
                " |V| = " + m_vValueFunction.size() +
                " backups = " + m_cBackups +
                " GComputations = " + AlphaVector.getGComputationsCount() +
                " #V^(b) = " + m_cNewPointComputations +
                " Dot products = " + AlphaVector.dotProductCount();
        Logger.getInstance().log("HSVI", 0, "VI", sMsg);

        if (ExecutionProperties.getReportOperationTime()) {
            sMsg = "Avg time: backup " + (m_cTimeInBackup / (m_cBackups * 1.0)) +
                    " G " + AlphaVector.getAvgGTime() +
                    " Tau " + m_pPOMDP.getBeliefStateFactory().getAvgTauTime() +
                    " DP " + AlphaVector.getAvgDotProductTime() +
                    " V^(b) " + (m_cTimeInV / (m_cNewPointComputations * 1.0) / 1000000) +
                    " HV(b) " + (m_cTimeInHV / (m_cApplyHComputations * 1.0));
            Logger.getInstance().log("HSVI", 0, "VI", sMsg);
        }
    }

    // ����bs������½�
    protected void updateBounds(BeliefState bsCurrent) {
        AlphaVector avNext = backup(bsCurrent);
        AlphaVector avCurrent = m_vValueFunction.getMaxAlpha(bsCurrent);
        double dCurrentValue = valueAt(bsCurrent);
        double dNewValue = avNext.dotProduct(bsCurrent);
        if (dNewValue > dCurrentValue) {
            m_vValueFunction.addPrunePointwiseDominated(avNext);
        }
        applyH(bsCurrent);
    }


    // explore��ȡ��һ�������
    protected BeliefState getNextBeliefState(BeliefState bsCurrent, double dEpsilon, double dDiscount) {
        //��ȡ���ŵ�action
        int iAction = getExplorationAction(bsCurrent);

        //����action��ȡ���ŵĹ۲�
        int iObservation = getExplorationObservation(bsCurrent, iAction, dEpsilon, dDiscount);

        if (iObservation == -1) {
            return null;
        }

        return bsCurrent.nextBeliefState(iAction, iObservation);
    }

    //explore����
    protected int explore(BeliefState bsCurrent, double dEpsilon, int iTime, double dDiscount, Vector<BeliefState> vObservedBeliefStates) {
        //��ʼ�����ֱ���
        double dWidth = width(bsCurrent);
        int iAction = 0, iObservation = 0;
        BeliefState bsNext = null;
        int iMaxDepth = 0;

        if (m_bTerminate)
            return iTime;

        // ���bs��֮ǰû��explore������ӵ�ObservedBeliefStates�㼯�ϡ�
        if (!vObservedBeliefStates.contains(bsCurrent))
            vObservedBeliefStates.add(bsCurrent);

        // ��¼���width
        if (dWidth > m_dMaxWidthForIteration)
            m_dMaxWidthForIteration = dWidth;

        // ��ȴ���200���߿��С����ֵ��epsilon/pow(gama,t))
        if (iTime > 200 || dWidth < (dEpsilon / dDiscount))
            return iTime;

        // �����һ�������
        bsNext = getNextBeliefState(bsCurrent, dEpsilon, dDiscount * m_dGamma);

        // ��һ����㲻Ϊ�� �� �����ڵ�ǰ�㣻�ݹ����explore����
        if ((bsNext != null) && (bsNext != bsCurrent)) {
            iMaxDepth = explore(bsNext, dEpsilon, iTime + 1, dDiscount * m_dGamma, vObservedBeliefStates);
        } else {
            iMaxDepth = iTime;
        }

        // ���µ�ǰ������½�
        updateBounds(bsCurrent);


        // ����������ʵ�ǰ�����explore
        // Ĭ�ϲ��������²�����
        if (m_dExplorationFactor > 0.0) {
            int iActionAfterUpdate = getExplorationAction(bsCurrent);
            if (iActionAfterUpdate != iAction) {
                if (m_rndGenerator.nextDouble() < m_dExplorationFactor) {
                    iObservation = getExplorationObservation(bsCurrent, iActionAfterUpdate, dEpsilon, dDiscount);
                    bsNext = bsCurrent.nextBeliefState(iAction, iObservation);
                    if (bsNext != null) {
                        iMaxDepth = explore(bsNext, dEpsilon, iTime + 1, dDiscount * m_dGamma, vObservedBeliefStates);
                        updateBounds(bsCurrent);
                    }
                }
            }
        }

        return iMaxDepth;
    }

    // ��ȡ���ŵĹ۲�
    protected int getExplorationObservation(BeliefState bsCurrent, int iAction,
                                            double dEpsilon, double dDiscount) {
        int iObservation = 0, iMaxObservation = -1;
        double dProb = 0.0, dExcess = 0.0, dValue = 0.0, dMaxValue = 0.0;
        BeliefState bsNext = null;

        for (iObservation = 0; iObservation < m_cObservations; iObservation++) {
            dProb = bsCurrent.probabilityOGivenA(iAction, iObservation);
            if (dProb > 0) {
                bsNext = bsCurrent.nextBeliefState(iAction, iObservation);
                dExcess = excess(bsNext, dEpsilon, dDiscount);
                dValue = dProb * dExcess;
                if (dValue > dMaxValue) {
                    dMaxValue = dValue;
                    iMaxObservation = iObservation;
                }
            }
        }
        return iMaxObservation;
    }

    protected int getExplorationAction(BeliefState bsCurrent) {
        /*ԭʼHSVI�㷨�������Ͻ�ȡaction*/
//		if (algorithmName == null) {
//			return m_vfUpperBound.getAction( bsCurrent );
//		}
//		
//		/*���ݷֲ�ȡaction*/
//		else {
//			return getActionByDistribution(bsCurrent);
//		}
//		long s1 = System.currentTimeMillis();
//		m_vfUpperBound.getAction( bsCurrent );
//		long e1 = System.currentTimeMillis();
//		long t1 = e1-s1;
//		
//		long s2 = System.currentTimeMillis();
//		getActionByDistribution(bsCurrent);
//		long e2 = System.currentTimeMillis();
//		long t2 = e2-s2;
//		
//		System.out.println("dis:"+t2+"\tori:"+t1+"\tdis-ori:"+(t2-t1));

        if (algorithmName == null) {
            return m_vfUpperBound.getAction(bsCurrent);
        }

        /*���ݷֲ�ȡaction*/
        else {
            return getActionByDistribution(bsCurrent);
        }


    }

    private int getActionByDistribution(BeliefState bs) {
        /*���ȷֲ��㷨*/
        /*��cMaxIteration�����飬ÿ���������һ�����е�action
         *��action���Ͻ���½죬���ݷֲ�ȡһ�����ֵ
         *��¼ȡֵ����action,��count+1
         *��cMaxIteration������֮��ȡcountֵ����action
         */
        int cMaxIteraiton = 1000;
        double[] upperBounds = new double[m_pPOMDP.getActionCount()];
        double[] lowerBounds = new double[m_pPOMDP.getActionCount()];

        for (int iAction = 0; iAction < m_pPOMDP.getActionCount(); iAction++) {
            upperBounds[iAction] = m_vfUpperBound.getValueByAction(bs, iAction);
//			System.out.println("up:"+upperBounds[iAction]);
            //lowerBounds[iAction] = G(iAction, bs, m_vValueFunction).dotProduct(bs);
            lowerBounds[iAction] = getLowerBound(bs, iAction, m_vValueFunction);
//			System.out.println("low:"+lowerBounds[iAction]);
        }

        int count[] = new int[m_pPOMDP.getActionCount()];
        for (int i = 0; i < cMaxIteraiton; i++) {
            int maxAction = 0;
            double maxValue = Double.NEGATIVE_INFINITY;
            for (int iAction = 0; iAction < m_pPOMDP.getActionCount(); iAction++) {
                // ���ȷֲ������½�ȡ���ֵ
                double upperBound = upperBounds[iAction];
                double lowerBound = lowerBounds[iAction];

                //double iValue = 0.0;
                double iValue = calculator.getValue(upperBound, lowerBound);
				/*if (algorithmName.equalsIgnoreCase("Avg")) {
					iValue = getValueByAverageDistribution(upperBound,lowerBound);
				}
				else if (algorithmName.equalsIgnoreCase("Tri")) {
					iValue = getValueByTriangleDistribution(upperBound, lowerBound);
				}
				else {
					iValue = getValueByBetaDistribution(upperBound, lowerBound);
				}*/
                //double iValue = getValueByTriangleDistribution(upperBound, lowerBound);
                //double iValue = getValueByBetaDistribution(upperBound, lowerBound);

                if (iValue > maxValue) {
                    maxValue = iValue;
                    maxAction = iAction;
                }
            }

            count[maxAction]++;
        }

        int bestAction = 0;
        int maxCount = 0;

        for (int iAction = 0; iAction < m_pPOMDP.getActionCount(); iAction++) {
            if (count[iAction] > maxCount) {
                bestAction = iAction;
                maxCount = count[iAction];
            }
        }

        return bestAction;
    }

    /**
     * ���ݾ��ȷֲ�ȡֵ
     * @param upperBound    �Ͻ�
     * @param lowerBound    �½�
     * @return
     */
	/*private double getValueByAverageDistribution(double upperBound, double lowerBound) {
		double width = upperBound - lowerBound;
		return m_rndGenerator.nextDouble(width) + lowerBound;
	}
	
	*//**
     * �������Ƿֲ�ȡֵ
     * @param upperBound    �Ͻ�
     * @param lowerBound    �½�
     * @return
     *//*
	private double getValueByTriangleDistribution(double upperBound, double lowerBound) {
		double width = upperBound - lowerBound;
		double rand_x = m_rndGenerator.nextDouble();
		
		//���Ƿֲ�����Ϊf(x)=2x,F(x)=pow(x,2), E(X)=2/3
//		return Math.pow(rand_x, 2) * width + lowerBound;
		
		//���Ƿֲ�����Ϊf(x)=2-2x, F(x) = 2x-pow(x,2) ,E(X)=1/3
		return (2*rand_x - Math.pow(rand_x, 2)) * width + lowerBound;
	}
	
	*/

    /**
     * ���ݣ�����ֲ�ȡֵ������a=1,b=3
     *
     * @param upperBound �Ͻ�
     * @param lowerBound �½�
     * @return
     *//*
	private double getValueByBetaDistribution(double upperBound, double lowerBound) {
		Beta b = new Beta();
		double width = upperBound - lowerBound;
		double rand_x = m_rndGenerator.nextDouble();
		return b.calculateBeta(1, 3, 0, rand_x)*width + lowerBound;
	}*/

    //��ȡ��ǰ�������½�ֵ��
    //��������backup����
    private double getLowerBound(BeliefState bs, int iAction, LinearValueFunctionApproximation vValueFunction) {
        // ��ʼ������ֵ��
		/*AlphaVector avMax = null, avG = null, avSum = null;
		List<AlphaVector> vVectors = new LinkedList<AlphaVector>(vValueFunction.getVectors());
		double dMaxValue = MIN_INF, dValue = 0, dProb = 0.0;
		
		// �������еĹ۲⣬
		for(int iObservation = 0 ; iObservation < m_cObservations ; iObservation++ ){
			dProb = bs.probabilityOGivenA( iAction, iObservation );
			if( dProb > 0.0 ){
				//dMaxValue = MIN_INF;
				//argmax_i g^i_a,o \cdot b
				//����ÿ�������Ļر�ֵ��ȡ�ر����������ĵ�˽����
				// ���½�
				for( AlphaVector avAlpha : vVectors ){
					if( avAlpha != null ){
						avG = avAlpha.G( iAction, iObservation );
						
						dValue = avG.dotProduct( bs );
						if( ( avMax == null ) || ( dValue >= dMaxValue ) ){
							dMaxValue = dValue;
							avMax = avG;
						}
					}
				}
			}
		}
		return dMaxValue;*/
        AlphaVector[] aNext = new AlphaVector[m_cObservations];
        return findMaxAlphas(iAction, bs, vValueFunction, aNext);
    }


    public class ValueFunctionEntry {
        private double m_dValue;
        private int m_iAction;
        private double[] m_adQValues;
        private int m_cActions;

        public ValueFunctionEntry(double dValue, int iAction) {
            m_dValue = dValue;
            m_iAction = iAction;
            m_cActions = m_pPOMDP.getActionCount();
            m_adQValues = new double[m_cActions];
            for (iAction = 0; iAction < m_cActions; iAction++) {
                m_adQValues[iAction] = Double.POSITIVE_INFINITY;
            }
        }

        public void setValue(double dValue) {
            m_dValue = dValue;
        }

        public double getValue() {
            return m_dValue;
        }

        public void setAction(int iAction) {
            m_iAction = iAction;
        }

        public int getAction() {
            return m_iAction;
        }

        public void setQValue(int iAction, double dValue) {
            m_adQValues[iAction] = dValue;
        }

        public double getQValue(int iAction) {
            return m_adQValues[iAction];
        }

        public double getMaxQValue() {
            double dMaxValue = Double.NEGATIVE_INFINITY;
            int iAction = 0;
            for (iAction = 0; iAction < m_cActions; iAction++) {
                if (m_adQValues[iAction] > dMaxValue)
                    dMaxValue = m_adQValues[iAction];
            }
            return dMaxValue;
        }

        public int getMaxAction() {
            double dMaxValue = Double.NEGATIVE_INFINITY;
            int iAction = 0, iMaxAction = -1;
            for (iAction = 0; iAction < m_cActions; iAction++) {
                if (m_adQValues[iAction] > dMaxValue) {
                    dMaxValue = m_adQValues[iAction];
                    iMaxAction = iAction;
                }
            }
            return iMaxAction;
        }
    }


    public DistributionCalculator getCalculator() {
        return calculator;
    }

    public void setCalculator(DistributionCalculator calculator) {
        this.calculator = calculator;
    }

}
