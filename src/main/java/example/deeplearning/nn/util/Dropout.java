
package example.deeplearning.nn.util;

import java.util.Random;

/**
 * Created by b1012059 on 2016/03/31.
 */
public class Dropout {

    public Dropout(){
    }

    /**
     * Dropout
     * @param size
     * @param p
     * @param rng
     * @return
     */
    public static int[] dropout(int size, double p, Random rng){
        int mask[] = new int[size];

        for(int i = 0; i < size; i++) {
            mask[i] = Distribution.binomial(1, p, rng);
        }

        return mask;
    }
}
