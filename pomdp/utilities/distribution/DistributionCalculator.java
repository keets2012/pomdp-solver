package pomdp.utilities.distribution;

public interface DistributionCalculator {
	/**
	 * ���ݷֲ�ȡֵ
	 * @param upperBound	�Ͻ�
	 * @param lowerBound	�½�
	 * @return
	 */
	public double getValue(double upperBound, double lowerBound);
}
