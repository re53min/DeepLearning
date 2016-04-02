package example.deeplearning.nn.layers;

import example.deeplearning.nn.util.ActivationFunction;
import example.deeplearning.nn.util.Distribution;

import java.util.Random;
import java.util.function.DoubleFunction;

/**
 * AutoEncoder(Denoising AutoEncoder)
 * Created by b1012059 on 2015/09/01.
 * @author Wataru Matsudate
 */
public class AutoEncoder {
    private int N;
    private int nIn;
    private int nOut;
    private Random rng;
    private double wIO[][];    //出力層の重み配列
    private double encodeBias[];    //出力層の閾値配列
    private double decodeBias[];    //decode用の各層配列
    private DoubleFunction<Double> activation;
    private DoubleFunction<Double> dActivation;

    /**
     * AutoEncoder Constructor
     * @param INPUT
     * @param OUTPUT
     */
    public AutoEncoder(int N, int INPUT, int OUTPUT, double wIO[][], double threshOut[],
                       Random rng, String activation){
        this.N = N;
        this.nIn = INPUT;
        this.nOut = OUTPUT;
        this.decodeBias = new double[INPUT];

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng  = rng;

        //入力層→出力層の重み配列をランダム
        if(wIO == null) {
            this.wIO = new double[this.nOut][this.nIn];
            //double element = 1.0 / nIn;
            for (int i = 0; i < this.nOut; i++) {
                for (int j = 0; j < this.nIn; j++) {
                    this.wIO[i][j] = Distribution.uniform(nOut, nIn, rng, activation);
                }
            }
        } else {
            this.wIO = wIO;
        }

        //入力層→出力層の閾値配列を0で初期化
        if(threshOut == null) this.encodeBias = new double[this.nOut];
        else this.encodeBias = threshOut;

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

    /**
     *ノイズ処理
     * 平均0、標準偏差1のガウス分布
     * @param x　Inputデータ
     * @param noiseX ノイズを付加したInputデータ
     * @param corruptionLevel 損傷率
     */
    public void noiseInput(double x[], double noiseX[], double corruptionLevel){

        for(int i = 0; i < x.length; i++) {
            if (x[i] == 0) noiseX[i] = 0;
            else noiseX[i] = x[i] + rng.nextGaussian() * corruptionLevel;
        }
    }

    /**
     * AutoEncoderにおける順方向計算(encode)の実現
     * 計算式:h = f(Wx+b)
     * なおf(x)はシグモイド関数
     * Wは重み行列、bはバイアス値
     * @param input
     * @param output
     */
    public void encoder (double input[], double output[]){
        int i,j;
        //各層の長さ
        int lengthIn = input.length;
        int lengthOut = output.length;

        //入力inputから符号outputを得る(encode)
        for(i = 0; i < lengthOut; i++){
            output[i] = 0;
            for(j = 0; j < lengthIn; j++){
                output[i] += input[j] * wIO[i][j];
            }
            output[i] += encodeBias[i];
            output[i] = activation.apply(output[i]);
        }
    }

    /**
     * AutoEncoderにおける逆方向計算(decode)の実現
     * 計算式:y = f'(W'h+b')
     * なおf'(x)はシグモイド関数、W'=W^T
     * W'は重み行列、b'はバイアス値
     * @param output
     * @param dInput
     */
    public void decoder(double output[] , double dInput[]){
        int i,j;
        //各層の長さ
        int lengthDeIn = output.length;
        int lengthDeOut = dInput.length;

        //符号outputから入力inputを復号する(decode)
        for(i = 0; i < lengthDeOut; i++) {
            dInput[i] = 0;
            for (j = 0; j < lengthDeIn; j++) {
                dInput[i] += output[j] * wIO[j][i];
            }
            dInput[i] += decodeBias[i];
            dInput[i] = activation.apply(dInput[i]);
        }
    }

    /**
     *
     * @param input
     * @param dinput
     */
    public void reconstruct(double input[], double dinput[]){
        double output[] = new double[nOut];

        encoder(input, output);
        decoder(output, dinput);
    }

    /**
     * trainメソッド
     * 確立的勾配降下法を用いてパラメータWの更新
     * @param x 学習データ
     * @param corruptionLevel 損傷率
     */
    public void train(double x[], double learningRate, double corruptionLevel){
        double noiseX[] = new double[nIn];
        double output[] = new double[nOut];
        double dInput[] = new double[nIn];

        noiseInput(x, noiseX, corruptionLevel);
        encoder(noiseX, output);
        decoder(output, dInput);

        //計算用
        double tempThO[] = new double[nOut];
        //誤差用配列
        double tempDeThO[] = new double[nIn];

        //閾値decodeThOの変更(decodeのbiasの変更)
        for(int i = 0; i < nIn; i++){
            tempDeThO[i] = x[i] - dInput[i];
            decodeBias[i] += learningRate * tempDeThO[i] / N;
        }

        //閾値threshOutの変更(encodeのbiasの変更)
        for(int i = 0; i < nOut; i++){
            tempThO[i] = 0;
            for(int j = 0; j < nIn; j++){
                tempThO[i] += wIO[i][j] * tempDeThO[j];
            }
            tempThO[i] *= dActivation.apply(output[i]);
            encodeBias[i] += learningRate * tempThO[i] / N;
        }

        //重みwIOの変更
        for(int i = 0; i < nOut; i++){
            for(int j  = 0; j < nIn; j++){
                wIO[i][j] += learningRate * (tempThO[i] * noiseX[j] + tempDeThO[j] * output[i]) / N;
            }
        }
    }

    public double[][] getWightIO(){
        return this.wIO;
    }

    public double[] getEncodeBias(){
        return this.encodeBias;
    }
}
