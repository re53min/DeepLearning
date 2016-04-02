package example.deeplearning.nn.layers;

import example.deeplearning.nn.util.Distribution;

import java.util.Random;

import static example.deeplearning.nn.util.utils.funSoftmax;

/**
 * Created by matsudate on 2016/03/04.
 */
public class SimpleLogisticRegression extends LogisticRegression {
    //private static Logger log = LoggerFactory.getLogger(LogisticRegression.class);
    private int nIn;
    private int nOut;
    public double wIO[][];
    private double bias[];
    private int N;
    private Random rng;

    /**
     * LogisticRegression Constructor
     *
     * @param nIn
     * @param nOut
     * @param N
     */
    public SimpleLogisticRegression(int nIn, int nOut, int N, Random rng) {
        super(nIn, nOut, N, rng);
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;
        this.wIO = new double[nOut][nIn];
        this.bias = new double[nOut];

        // log.info("Initialize LogisticLayer");

        //ランダムの種
        if (rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //重み行列の初期化
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                wIO[i][j] = Distribution.uniform(nIn, nOut, rng, null);
            }
        }

        //バイアスの初期化
        for (int i = 0; i < nOut; i++) {
            bias[i] = 0;
        }
    }

    /**
     * Training Method
     *
     * @param input
     * @param teach
     * @param learningRate
     */
    public double[] train(double input[], int teach[], double learningRate) {
        double output[] = new double[nOut];
        double dOutput[] = new double[nOut];

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
        //Softmax関数
        funSoftmax(output, nOut);

        /*
        ロジスティック回帰の逆方向学習
        確率的勾配降下法を用いてパラメータ更新
         */
        for (int i = 0; i < nOut; i++) {
            //教師信号との誤差を求める
            dOutput[i] = teach[i] - output[i];

            for (int j = 0; j < nIn; j++) {
                //重み行列の更新
                wIO[i][j] += learningRate * dOutput[i] * input[j] / N;
            }
            //バイアスの更新
            bias[i] += learningRate * dOutput[i] / N;
        }
        return dOutput;
    }

    /**
     * Testing Data Method
     *
     * @param input  テストデータ
     * @param output 出力
     */
    public void reconstruct(double input[], double output[]) {
        //学習されたパラメータを使用した順方向計算
        for (int i = 0; i < nOut; i++) {
            //初期化
            output[i] = 0;
            for (int j = 0; j < nIn; j++) {
                output[i] += input[j] * wIO[i][j];
            }
            output[i] += bias[i];
        }

        //Softmax関数
        funSoftmax(output, nOut);

    }
}
