package example.deeplearning.nn.util;

import java.util.Random;

/**
 * Created by b1012059 on 2016/03/31.
 */
public class Distribution {

    public Distribution(){
    }

    /**
     * 一様分布
     * @param nIn
     * @param nOut
     * @param rng
     * @param activation
     * @return
     */
    public static double uniform(int nIn, int nOut, Random rng, String activation){
        double min = -Math.sqrt(6. / (nIn + nOut));
        double max = Math.sqrt(6. / (nIn + nOut));

        if(activation == "sigmoid" || activation == null) {
            min *= 4;
            max *= 4;
        }
        return rng.nextDouble() * (max - min) + min;
    }

    /**
     * 一様分布
     * @param nIn
     * @param nOut
     * @param rng
     * @return
     */
    public static double uniform(double nIn, double nOut, Random rng){
        return rng.nextDouble() * (nOut - nIn) + nIn;
    }

    /**
     * 二項分布
     * @param n
     * @param p
     * @param rng
     * @return
     */
    public static int binomial(int n, double p, Random rng) {
        if(p < 0 || p > 1) return 0;

        int c = 0;
        double r;

        for(int i = 0; i < n; i++) {
            r = rng.nextDouble();
            if (r < p) c++;
        }

        return c;
    }
}
