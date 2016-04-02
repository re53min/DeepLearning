package example.deeplearning.nn.util;

/**
 * Created by b1012059 on 2015/09/08.
 */
public class utils {

    //private static Logger log = LoggerFactory.getLogger(utils.class);

    public utils(){
    }


    /**
     * ソフトマックス関数
     * @param tmpOut
     * @param nOut
     * @return
     */
    public static void funSoftmax(double tmpOut[], int nOut){
        double max = 0.0;
        double sum = 0.0;

        for(int i = 0; i < nOut; i++){
            if(max < tmpOut[i]) max = tmpOut[i];
        }

        for(int i = 0; i < nOut; i++){
            tmpOut[i] = Math.exp(tmpOut[i] - max);
            tmpOut[i] = Math.exp(tmpOut[i]);
            sum += tmpOut[i];
        }

        for(int i = 0; i < nOut; i++){
            tmpOut[i] /= sum;
        }
    }

    /**
     *
     * @param learningRate
     * @param decayRate
     * @param epoch
     * @return
     */
    public static double updateLR(double learningRate, double decayRate, int epoch){
        return learningRate / (1 + decayRate * epoch);
    }

    /**
     * AdaGradの実装
     * @param learningRate
     * @param prevRT
     * @param error
     * @param ada
     * @return
     */
    public static double adaGrad(double learningRate, double prevRT, double error, double ada){
        double tmpLearningRate = prevRT + Math.pow(error, 2);
        return learningRate / Math.sqrt(tmpLearningRate + ada);
    }

    /**
     * RMSPropの実装
     * @param learningRate
     * @param prevRT
     * @param hyperP
     * @param error
     * @param rms
     * @return
     */
    public static double rmsProp(double learningRate, double prevRT, double hyperP, double error, double rms){
        double tmpLearningRate = hyperP*prevRT + (1-hyperP)*Math.pow(error, 2);
        return learningRate / Math.sqrt(tmpLearningRate + rms);
    }

    /**
     * コサイン類似度
     * cosθ = (vectorA*vectorB) / (normA*normB)
     * 分子はベクトルの内積、分母はそれぞれのノルム
     * @param vectorA
     * @param vectorB
     * @return
     */
    public static double cosineSimilarity(double[] vectorA, double[] vectorB){
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for(int i = 0; i < vectorA.length; i++){
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }

        return dotProduct /(Math.sqrt(normA) * Math.sqrt(normB));
    }
}
