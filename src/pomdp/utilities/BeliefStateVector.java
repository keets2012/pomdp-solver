package pomdp.utilities;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;

/**
 * ��ͨ�㷨Ҳ�����ճ�ʹ�� �ڵ�����֮ǰ�Ѿ�ȷ����b�����������
 *
 * @param <E>
 * @author default
 */
public class BeliefStateVector<E> extends Vector<E> {
    /**
     *
     */
    private static final long serialVersionUID = 2174749846857971924L;

    /*
     * 0���� 1����һ�� 2���ڶ���
     */
    private ArrayList<ArrayList<E>> treeLevelInfo = null;

    /*
     * ���ֻ����µ�b����ô������tree��Ϣ���ͷ�����
     */
    private ArrayList<ArrayList<E>> treeLevelInfoComplete = null;
    private boolean notComplete = false;

    public BeliefStateVector() {
        super();
        treeLevelInfo = new ArrayList<ArrayList<E>>();
    }

    public BeliefStateVector(Vector<E> vector) {
        super(vector);
    }

    public BeliefStateVector(BeliefStateVector<E> beliefStateVector) {
        super(beliefStateVector);
        ArrayList<ArrayList<E>> treeLevelInfoIn = beliefStateVector.getTreeLevelInfo();
        treeLevelInfo = new ArrayList<ArrayList<E>>();
        for (int i = 0; i < treeLevelInfoIn.size(); i++) {
            treeLevelInfo.add(new ArrayList<E>());
            treeLevelInfo.get(i).addAll(treeLevelInfoIn.get(i));
        }

    }

    /**
     * ���ֻ����µ�b����ô������tree��Ϣ����Ҫ����洢
     *
     * @param beliefStateVector
     * @param notComplete
     */
    public BeliefStateVector(BeliefStateVector<E> beliefStateVector,
                             boolean notCompleteIn) {
        this();
        ArrayList<ArrayList<E>> treeLevelInfoIn = beliefStateVector.getTreeLevelInfo();
        treeLevelInfoComplete = new ArrayList<ArrayList<E>>();
        for (int i = 0; i < treeLevelInfoIn.size(); i++) {
            treeLevelInfoComplete.add(new ArrayList<E>());
            treeLevelInfoComplete.get(i).addAll(treeLevelInfoIn.get(i));
        }
        for (int i = 0; i < treeLevelInfoComplete.size(); i++) {
            treeLevelInfo.add(new ArrayList<E>());
        }
        notComplete = true;
    }

    /**
     * ����һ��bʱ������parent��Ϣ ��Ϊ������parentΪnull ���b�Ѿ����ڣ���������
     *
     * @param parent
     * @param current
     * @return
     */
    public synchronized void add(E parent, E current) {
        super.add(current);
        // Ϊ��
        if (parent == null) {
            treeLevelInfo.clear();
            ArrayList<E> levelOne = new ArrayList<E>();
            treeLevelInfo.add(levelOne);
            levelOne.add(current);
        }
        // �Ǹ�
        else {
            int pLevelNum = getLevelNum(parent);
            if (pLevelNum < 0) {
                Logger.getInstance()
                        .logln("Error: BeliefStateVector.add(E parent, E current): parent not exist!");
                return;
            }
            // parent����
            else {
                // parent����һ�����current
                pLevelNum += 1;
                if (treeLevelInfo.size() <= pLevelNum) {
                    ArrayList<E> newLevel = new ArrayList<E>();
                    treeLevelInfo.add(newLevel);
                    if (notComplete) {
                        ArrayList<E> newLevel2 = new ArrayList<E>();
                        treeLevelInfoComplete.add(newLevel2);
                    }
                }
                treeLevelInfo.get(pLevelNum).add(current);
                if (notComplete)
                    treeLevelInfoComplete.get(pLevelNum).add(current);
            }
        }
    }

    /**
     * Ĭ�ϵ�������������϶�����notComplete==true
     * Ĭ�������ϲ�֮�󣬾�����������
     */
    @SuppressWarnings("unchecked")
    @Override
    public synchronized boolean addAll(Collection<? extends E> bsv) {
        // Ϊ��֧����ͨvector�Ĺ��ܣ�������һ���ж�
        if (bsv instanceof BeliefStateVector<?>) {
            ArrayList<ArrayList<E>> treeLevelInfoIn = ((BeliefStateVector<E>) bsv)
                    .getTreeLevelInfo();
            for (int i = 0; i < treeLevelInfoIn.size(); i++) {
                if (treeLevelInfo.size() <= i) {
                    treeLevelInfo.add(new ArrayList<E>());
                }
                treeLevelInfo.get(i).addAll(treeLevelInfoIn.get(i));

            }
            notComplete = false;
        }
        return super.addAll(bsv);
    }

    /**
     * ��ѯһ��b�Ĳ���
     *
     * @param e
     * @return -1: not found
     */
    public synchronized int getLevelNum(E e) {
        if (notComplete) {
            for (int i = 0; i < treeLevelInfoComplete.size(); i++) {
                if (treeLevelInfoComplete.get(i).contains(e))
                    return i;
            }
        } else {
            for (int i = 0; i < treeLevelInfo.size(); i++) {
                if (treeLevelInfo.get(i).contains(e))
                    return i;
            }
        }
        return -1;
    }

    public synchronized ArrayList<ArrayList<E>> getTreeLevelInfo() {
        return treeLevelInfo;
    }

    public synchronized ArrayList<ArrayList<E>> getTreeLevelInfoComplete() {
        return treeLevelInfoComplete;
    }

    /**
     * ���һ����tree�ϣ���������ɨ���iterator
     *
     * @return
     */
    public Iterator<E> getTreeDownUpIterator() {
        ArrayList<E> resultList = new ArrayList<E>();
        for (int i = treeLevelInfo.size() - 1; i >= 0; i--) {
            resultList.addAll(treeLevelInfo.get(i));
        }
        return resultList.iterator();
    }

    /**
     * �϶���complete��
     * �ر�ķ�����Ϊ�˴�Vector��ɾ��b��ͬʱ����tree��Ҳɾ����
     */
    public void removeABeliefStateWithTree(E e) {
        super.remove(e);
        for (int i = 0; i < treeLevelInfo.size(); i++) {
            if (treeLevelInfo.get(i).contains(e))
                treeLevelInfo.get(i).remove(e);
        }
    }
}
