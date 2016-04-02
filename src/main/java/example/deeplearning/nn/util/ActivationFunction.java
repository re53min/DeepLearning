package example.deeplearning.nn.util;

/**
 * Created by b1012059 on 2016/04/02.
 */
public class ActivationFunction {

    public ActivationFunction(){
    }

    //シグモイド関数の傾き
    private final static double beta = 1.0;


    /**
     *シグモイド関数
     * @param tmpOut
     * @return sigmoid
     */
    public static double funSigmoid(double tmpOut){
        double sigmoid = 1.0 / (1.0 + Math.exp(-beta * tmpOut));
        return sigmoid;
    }

    /**
     * シグモイド関数の微分
     * @param tmpOut
     * @return dsigmoid
     */
    public static double dfunSigmoid(double tmpOut){
        double dsigmoid = tmpOut * (1 - tmpOut);
        return dsigmoid;
    }

    /**
     *
     * @param tmpOut
     * @return
     */
    public static double funTanh(double tmpOut){
        double tanh = Math.tanh(tmpOut);
        return tanh;
    }

    /**
     *
     * @param tmpOut
     * @return
     */
    public static double dfunTanh(double tmpOut){
        double dtanh = 1 - tmpOut * tmpOut;
        return dtanh;
    }

    /**
     * Relu(ランプ関数)
     * @param tmpOut
     * @return
     */
    public static double funReLU(double tmpOut){

        if(tmpOut > 0) return tmpOut;
        else return  0.;
    }

    /**
     * Relu(ランプ関数)の微分
     * @param tmpOut
     * @return
     */
    public static double dfunReLU(double tmpOut){

        if(tmpOut > 0) return 1.;
        else return 0.;
    }
}
