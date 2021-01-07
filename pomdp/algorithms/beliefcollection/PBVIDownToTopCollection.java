package pomdp.algorithms.beliefcollection;
import java.util.Vector;

import pomdp.algorithms.ValueIteration;
import pomdp.utilities.BeliefState;
import pomdp.utilities.BeliefStateVector;

public class PBVIDownToTopCollection extends BeliefCollection{

	boolean fullSuccessors;
	
	public PBVIDownToTopCollection(ValueIteration vi, boolean _fullSuccessors, boolean bAllowDuplicates)
	{	
		super(vi, bAllowDuplicates);	
		fullSuccessors = _fullSuccessors;
	}
	
	
	/* in PBVI, we start only with the initial belief state of the POMDP */
	public Vector<BeliefState> initialBelief()
	{
		
		BeliefStateVector<BeliefState> initial = new BeliefStateVector<BeliefState>();
		
		/* initialize the list of belief points with the initial belief state */
		initial.add(null, POMDP.getBeliefStateFactory().getInitialBeliefState() );
		
		return initial;
	}
	
	
	
	/**
	 * �÷���������tree���޸�
	 */
	public Vector<BeliefState> expand(int numNewBeliefs, Vector<BeliefState> beliefPointsIn)
	{
		BeliefStateVector<BeliefState> beliefPoints = (BeliefStateVector<BeliefState>)beliefPointsIn;
		BeliefStateVector<BeliefState> newBeliefs = new BeliefStateVector<BeliefState>(beliefPoints, true);			
		BeliefStateVector<BeliefState> combinedBeliefs = new BeliefStateVector<BeliefState>(beliefPoints);
		BeliefState picked;
		
		while (newBeliefs.size() < numNewBeliefs && combinedBeliefs.size() > 0 )
		{
			int randomBelief = valueIteration.getPOMDP().getRandomGenerator().nextInt(combinedBeliefs.size());
			picked = combinedBeliefs.get(randomBelief);
			
			BeliefState bsNext;
			if (fullSuccessors)
				 bsNext = POMDP.getBeliefStateFactory().computeFarthestSuccessorFull(combinedBeliefs, picked);
			else
				 bsNext = POMDP.getBeliefStateFactory().computeFarthestSuccessor(combinedBeliefs, picked);
			
			//��b�ĺ�̵㶼��B���ˣ��Ͱ�b��B��ȥ��
			if( bsNext == null )//do not choose again a belief who has all its successors already in B
				combinedBeliefs.remove( picked );
//				combinedBeliefs.removeABeliefStateWithTree(picked);
			
			if ((bsNext != null) && (!combinedBeliefs.contains(bsNext))) {
				newBeliefs.add(picked, bsNext);
				combinedBeliefs.add(picked, bsNext);
				
			}		
					
		}
		return newBeliefs;	
	}
	
	
	/**
	 * �÷���������tree���޸�
	 */
	public Vector<BeliefState> expand(Vector<BeliefState> beliefPoints){
			
		BeliefStateVector<BeliefState> newBeliefs = new BeliefStateVector<BeliefState>((BeliefStateVector<BeliefState>)beliefPoints, true);			
		BeliefState bsNext;
		
		for (BeliefState bsCurrent : beliefPoints)
		{
			bsNext = POMDP.getBeliefStateFactory().computeFarthestSuccessor(beliefPoints, bsCurrent);
			if( (bsNext != null) && (!newBeliefs.contains(bsNext)) && (!beliefPoints.contains(bsNext))){
				//����parent��Ϣ
				newBeliefs.add(bsCurrent, bsNext);
			}		
		}	
		return newBeliefs;
	}
	
}
