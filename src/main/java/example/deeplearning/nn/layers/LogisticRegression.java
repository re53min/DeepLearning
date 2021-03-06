package example.deeplearning.nn.layers;

import example.deeplearning.nn.util.utils;
import example.deeplearning.nn.util.Distribution;

import java.util.Random;

/**
 * Logistic Regression
 * Created by b1012059 on 2015/11/23.
 */
public class LogisticRegression {
    //private static Logger log = LoggerFactory.getLogger(LogisticRegression.class);
    private int nIn;
    private int nOut;
    private int dim;
    private double wIO[][];
    private double wPO[][];
    private double bias[];
    private int N;
    private Random rng;

    /**
     * LogisticRegression Constructor
     * @param nIn
     * @param nOut
     * @param N
     */
    public LogisticRegression(int nIn, int nOut, int N, Random rng){
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;
        this.wIO = new double[nOut][nIn];
        this.bias = new double[nOut];

       // log.info("Initialize LogisticLayer");

        //ランダムの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //重み行列の初期化
        for(int i = 0; i < nOut; i++) {
            for(int j = 0; j < nIn; j++){
                wIO[i][j] = Distribution.uniform(nIn, nOut, rng, null);
            }
        }
    }

    /**
     *
     * @param dim
     * @param nIn
     * @param nOut
     * @param N
     * @param rng
     * @param activation
     */
    public LogisticRegression(int dim, int nIn, int nOut, int N, Random rng, String activation){
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;
        this.dim = dim;
        this.wIO = new double[nOut][nIn];
        this.wPO = new double[nOut][dim];
        this.bias = new double[nOut];

        //log.info("Initialize LogisticLayer");

        //ランダムの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //重み行列の初期化
        for(int i = 0; i < nOut; i++) {
            for(int j = 0; j < nIn; j++){
                wIO[i][j] = Distribution.uniform(nIn, nOut, rng, activation);
            }
        }
    }

    /**
     * Training Method
     * @param input
     * @param teach
     * @param learningRate
     * @return
     */
    public double[] train(double input[], int teach[], double learningRate){
        double output[] = new double[nOut];
        double dOutput[] = new double[nOut];

        /*
        ロジスティック回帰の順方向計算
         */
        for(int i = 0; i < nOut; i++){
            output[i] = 0;
            for(int j = 0; j < nIn; j++){
                //入力とそれに対する重み行列の積
                output[i] += wIO[i][j] * input[j];
            }
            //バイアス
            output[i] += bias[i];
        }
        //Softmax関数
        utils.funSoftmax(output, nOut);

        /*
        ロジスティック回帰の逆方向学習
        確率的勾配降下法を用いてパラメータ更新
         */
        for(int i = 0; i < nOut; i++){
            //教師信号との誤差を求める
            dOutput[i] = teach[i] - output[i];

            for(int j = 0; j < nIn; j++){
                //重み行列の更新
                wIO[i][j] += learningRate * dOutput[i] * input[j] / N;
            }
            //バイアスの更新
            bias[i] += learningRate * dOutput[i] / N;
        }

        return dOutput;
    }

    public void train2(double input[], double projection[][], int teach[],
                       double dProjection[][], double dhOutput[], double learningRate){

        double output[] = new double[nOut];
        double dOutput[] = new double[nOut];

        /*
        ロジスティック回帰の順方向計算
         */
        for(int i = 0; i < nOut; i++){
            output[i] = 0;
            for(int j = 0; j < nIn; j++){
                //入力とそれに対する重み行列の積
                output[i] += wIO[i][j] * input[j];
            }
            //バイアス
            output[i] += bias[i];
        }

        for(int n = 0; n < projection.length; n++) {
            for (int i = 0; i < nOut; i++) {
                for (int j = 0; j < dim; j++) {
                    output[i] += wPO[i][j] * projection[n][j];
                }
            }
        }

        //Softmax関数
        //log.info("Softmax Function:");
        utils.funSoftmax(output, nOut);

        for(int z = 0; z < nOut; z++) {
            try {
                if (Double.isInfinite(output[z])) {
                    throw new Exception("OutputがInfinityになりました");
                } else if (Double.isNaN(output[z])) {
                    throw new Exception("OutputがNaNになりました");
                } else if (output[z] >= Double.MAX_VALUE) {
                    throw new Exception("Outputがオーバーフローしました");
                } else if (output[z] <= Double.MIN_VALUE) {
                    throw new Exception("Outputがアンダーフローしました");
                }
            } catch (Exception e){
                e.printStackTrace();
            }
        }
        /*
        for(int i = 0; i < nOut; i++) {
            System.out.print(output[i] + " ");
        }
        System.out.println();
        */

        /*
        ロジスティック回帰の逆方向学習
        確率的勾配降下法を用いてパラメータ更新
         */
        for (int i = 0; i < nOut; i++) {
            //教師信号との誤差を求める
            dOutput[i] = teach[i] - output[i];

            for (int j = 0; j < nIn; j++) {
                //中間層→出力層の誤差勾配
                dhOutput[j] += dOutput[i] * wIO[i][j];
                //中間層→出力層の重み行列更新
                wIO[i][j] += learningRate * dOutput[i] * input[j];
            }

            //バイアスの更新
            bias[i] += learningRate * dOutput[i];

            for(int n = 0; n < projection.length; n++) {
                for (int k = 0; k < dim; k++) {
                    //投影層→出力層の誤差勾配
                    dProjection[n][k] += dOutput[i] * wPO[i][k];

                    //投影層→出力層の重み行列更新
                    wPO[i][k] += learningRate * dOutput[i] * projection[n][k];
                }
            }
        }
    }

    /**
     * Testing Data Method
     * @param input テストデータ
     * @param output 出力
     */
    public void reconstruct(double input[], double output[]){
        //学習されたパラメータを使用した順方向計算
        for(int i = 0; i < nOut; i++){
            //初期化
            output[i] = 0;
            for(int j = 0; j < nIn; j++){
                output[i] += input[j] * wIO[i][j];
            }
            output[i] += bias[i];
        }

        //Softmax関数
        utils.funSoftmax(output, nOut);

    }

    public void reconstruct2(double input[], double projection[][], double output[]) {
        //学習されたパラメータを使用した順方向計算
        /*
        ロジスティック回帰の順方向計算
         */
        for (int i = 0; i < nOut; i++) {
            output[i] = 0;
            for (int j = 0; j < nIn; j++) {
                //入力とそれに対する重み行列の積
                output[i] += wIO[i][j] * input[j];
            }
            //バイアス
            output[i] += bias[i];
        }

        for (int n = 0; n < projection.length; n++) {
            for (int i = 0; i < nOut; i++) {
                for (int j = 0; j < dim; j++) {
                    output[i] += wPO[i][j] * projection[n][j];
                }
            }
        }

        //Softmax関数
        //log.info("Softmax Function:");
        utils.funSoftmax(output, nOut);
    }

    public double[][] getW(){
        return this.wIO;
    }

    public double[][] getWPO(){
        return this.wPO;
    }
}
