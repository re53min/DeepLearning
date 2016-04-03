package example.deeplearning.nn.multilayer;

import example.deeplearning.nn.layers.HiddenLayer;
import example.deeplearning.nn.layers.LogisticRegression;

import java.util.Random;

/**
 * バックプロパゲーションの練習問題3
 * API化したHiddenLayer及びLogisticRegressionを用いたBackPropagationの実現
 * テストデータがどの数字に一番近いかの分類
 * Created by b1012059 on 2015/11/22.
 */
public class MultiLayerPerceptron {
    //private static Logger log = LoggerFactory.getLogger(MultiLayerPerceptron.class);
    private int nInput;
    private int hiddenSize;
    private int nOutput;
    private int N;
    private HiddenLayer hLayer;
    private LogisticRegression logisticLayer;
    private Random rng;

    public MultiLayerPerceptron(int INPUT, int HIDDEN, int OUTPUT, int N, Random rng, String activation){

        this.N = N;
        this.hiddenSize = HIDDEN;
        this.nInput = INPUT;
        this.nOutput = OUTPUT;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        this.hLayer = new HiddenLayer(this.nInput, this.hiddenSize, null, null, N, rng, activation);
        this.logisticLayer = new LogisticRegression(this.hiddenSize, this.nOutput, N, rng);
    }

    /**
     * Training Method
     * @param input 入力データ
     * @param teach 教師データ
     * @param learningRate 学習率
     */
    public void train(double input[][], int teach[][], double learningRate){
        double[] hiddenInput;
        double[] outLayerInput;
        double[] dOutput;


        for(int n = 0; n < N; n++) {
            hiddenInput = new double[nInput];
            outLayerInput = new double[hiddenSize];

            for(int j = 0; j < nInput; j++) hiddenInput[j] = input[n][j];
            hLayer.forwardCal(hiddenInput, outLayerInput);
            dOutput = logisticLayer.train(outLayerInput, teach[n], learningRate);

            hLayer.backwardCal(hiddenInput, null, outLayerInput, dOutput, logisticLayer.getW(), learningRate);
        }
    }

    /**
     * Testing Data Method
     * @param input
     * @param output
     */
    public void reconstruct(double input[], double output[]){
        double outLayerInput[] = new double[hiddenSize];

        hLayer.forwardCal(input, outLayerInput);
        logisticLayer.reconstruct(outLayerInput, output);
    }
}
