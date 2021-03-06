package example.deeplearning.nn.layers;

import example.deeplearning.nn.util.ActivationFunction;
import example.deeplearning.nn.util.Distribution;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;
import java.util.function.DoubleFunction;

/**
 * Created by b1012059 on 2016/02/15.
 */
public class RecurrentHLayer {
    private static Logger log = LoggerFactory.getLogger(HiddenLayer.class);
    private int nIn;
    private int nOut;
    private double wIH[][];
    private double wRH[][];
    private double bIH[];
    private double bRH[];
    private int N;
    private Random rng;
    private DoubleFunction<Double> activation;
    private DoubleFunction<Double> dActivation;

    public RecurrentHLayer(int nIn, int nOut, double wIH[][], double wRH[][],
                           double bIH[], double bRH[], int N, Random rng, String activation){
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;

        log.info("Initialize HiddenLayer");

        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //重み行列を連続一様分布で初期化
        if(wIH == null){
            this.wIH = new double[nOut][nIn];
            for(int i = 0; i < nOut; i++){
                for(int j = 0; j < nIn; j++){
                    this.wIH[i][j] = Distribution.uniform(nIn, nOut, rng, activation);
                }
            }
        } else{
            this.wIH = wIH;
        }

        if(wRH == null){
            this.wRH = new double[nOut][nOut];
            for(int i = 0; i < nOut; i++){
                for(int j = 0; j < nOut; j++){
                    this.wRH[i][j] = Distribution.uniform(nOut, nOut, rng, activation);
                }
            }
        } else {
            this.wRH = wRH;
        }

        //バイアスを0で初期化
        if(bIH == null && bRH == null){
            this.bIH = new double[nOut];
            this.bRH = new double[nOut];
        } else {
            this.bIH = bIH;
            this.bRH = bRH;
        }

        /*
        ここラムダ式で記述
         */
        if (activation == "sigmoid" || activation == null) {
            this.activation = (double tmpOut) -> ActivationFunction.funSigmoid(tmpOut);
            this.dActivation = (double tmpOut) -> ActivationFunction.dfunSigmoid(tmpOut);
        } else if(activation == "tanh"){
            this.activation = (double tmpOut) -> ActivationFunction.funTanh(tmpOut);
            this.dActivation = (double tmpOut) -> ActivationFunction.dfunTanh(tmpOut);
        } else if(activation == "ReLU"){
            this.activation = (double tmpOut) -> ActivationFunction.funReLU(tmpOut);
            this.dActivation = (double tmpOut) -> ActivationFunction.dfunReLU(tmpOut);
        } else {
            //log.info("Activation function not supported!");
        }
    }

    public double[] lookUpTable(int wordToId){
        double[] lookUp = new double[nIn];

        for(int j = 0; j < nOut; j++){
            lookUp[j] = wIH[j][wordToId];
        }
        return lookUp;
    }

    public void forwardCal(int wordToId, double rInput[], double output[]){
        for(int i = 0; i < nOut; i++) {
            output[i] = this.output(wordToId, rInput, wRH[i], bIH[i], bRH[i]);
        }
    }

    private double output(int wordToId, double rInput[],
                          double r[], double bIH, double bRH){
        double tmpData = 0.0;
        for(int i = 0; i < nIn; i++){
            tmpData += this.lookUpTable(wordToId)[i];
        }
        tmpData += bIH;

        for(int j = 0; j < nOut; j++){
            tmpData += rInput[j] * r[j];
        }
        tmpData += bRH;

        return activation.apply(tmpData);
    }

    /**
     * Backward Calculation
     * @param wordToId 今層への入力(前層の出力)
     * @param dOutput 今層の誤差勾配
     * @param prevInput 次層への入力(今層の出力)
     * @param prevdOutput　次層の誤差勾配
     * @param prevWIO 次層の重み行列(今層→次層への重み行列)
     * @param learningRate 学習率
     */
    public void backwardCal(int wordToId, double dOutput[], double prevInput[],
                            double prevdOutput[], double prevWIO[][], double rInput[], double learningRate){

        if(dOutput == null) dOutput = new double[nOut];

        //今層の誤差勾配
        for(int i = 0; i < nOut; i++) {
            dOutput[i] = 0;
            for (int j = 0; j < prevdOutput.length; j++) {
                //次層の誤差勾配と次層への重み行列との積
                dOutput[i] += prevdOutput[j] * prevWIO[j][i];
            }
            //活性化関数の微分との積により誤差勾配(修正量)を求める
            dOutput[i] *= dActivation.apply(prevInput[i]);
        }

        //今層の誤差勾配を用いて重み行列及びバイアスの更新
        for(int i = 0; i < nOut; i++){
            for(int j = 0; j < nIn; j++){
                //重み行列の更新
                wIH[i][j] += learningRate * dOutput[i] * this.lookUpTable(wordToId)[j] / N;
            }

            for(int k = 0; k < nOut; k++){
                //再帰
                wRH[i][k] += learningRate * rInput[k] * dOutput[i] / N;
            }

            //バイアスの更新
            bIH[i] += learningRate * dOutput[i] / N;
            bRH[i] += learningRate * dOutput[i] / N;
        }
    }

    public double[][] getwIH(){
        return this.wIH;
    }

    /*
    ここにDropOut
     */
    public void dropout(){

    }
}
