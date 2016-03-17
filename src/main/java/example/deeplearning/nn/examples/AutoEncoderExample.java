package example.deeplearning.nn.examples;

import example.deeplearning.imageprocessing.ImageProcessing;
import example.deeplearning.nn.layers.AutoEncoder;

import java.util.Random;

/**
 * Created by b1012059 on 2016/03/12.
 */
public class AutoEncoderExample {

    public static void main(String args[]){

        //入力データ
        double inputData[][] = {

                //0
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //1
                {0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  1,  1,  0,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //2
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //3
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //4
                {0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  1,  0,  0,  0,
                        0,  1,  0,  1,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //5
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //6
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //7
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  1,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //8
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //9
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  0,  0}
        };

        //testデータ
        double testData[][] = {
                //1
                {0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  1,  1,  0,  0,
                        0,  0,  0,  0,  0,  0,  0}

        };


        int nIn = 63;
        int nOut = 10;
        int inputSize = inputData.length;
        Random r = new Random(123);
        int epochs = 400;
        double alpha = 0.1;
        double corruptionLevel = 0.3;

        //インスタンスの生成
        AutoEncoder ae = new AutoEncoder(inputSize, nIn, nOut, null, null, r, null);

        //train
        for(int epoch = 0; epoch < epochs; epoch++){
            for (int i = 0; i < inputData.length; i++) {
                ae.train(inputData[i], alpha, corruptionLevel);
            }
        }

        //weight output
        System.out.println("------------------------------------------------------");
        ImageProcessing img = new ImageProcessing();
        for(int i = 0; i < nIn; i++) {
            img.imagePlot(ae.getWightIO()[i], 25, 25);
        }

        /*
        for(int i = 0; i < nOut; i++){
            for(int j = 0; j < nIn; j++){
                System.out.printf("%.5f", ae.getWightIO()[i][j] + " ");
            }
            System.out.println("");
        }
        */
        System.out.println("------------------------------------------------------");


        //test
        double reconstructedX[][] = new double[testData.length][nIn];
        System.out.println("Test AutoEncoder");
        for(int i = 0; i < testData.length; i++){
            ae.reconstruct(testData[i], reconstructedX[i]);
            for(int j = 0; j < nIn; j++){
                System.out.printf("%.5f ", reconstructedX[i][j]);
            }
            System.out.println();
        }
    }
}
