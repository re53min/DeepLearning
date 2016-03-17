package example.deeplearning.nn.examples;

import example.deeplearning.nn.multilayer.StackedAutoEncoder;

import java.util.Random;

/**
 * Created by b1012059 on 2016/03/12.
 */
public class StackedAutoEncoderExample {

    public static void main(String args[]){

        //入力データ
        int inputData[][] = {
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
                /*{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}*/
        };

        //教師データ
        int teachData[][] = {
                {1, 0},
                {1, 0},
                {1, 0},
                {0, 1},
                {0, 1},
                {0, 1}
        };

        //testデータ
        int testData[][] = {
                {1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
                /*
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                 */
        };

        int nInput = 10;
        int nHidden[] = {8, 6, 4};
        int nOutput = 2;
        Random rng = new Random(123);
        int epoch = 1000;
        double corruptionLevel = 0.3;
        double alpha = 0.1;
        double decayRate = 1E-2;

        //インスタンスの生成
        StackedAutoEncoder sAE = new StackedAutoEncoder(nInput, nHidden, nOutput,
                inputData.length, rng, "ReLU");

        //pre-training
        sAE.preTraining(inputData, alpha, epoch, corruptionLevel);
        //fine-tuning
        sAE.fineTuning(inputData, teachData, alpha, epoch, decayRate);

        //test
        int nTest = 2;
        double testOut[][] = new double[nTest][nOutput];

        //Output
        for(int i = 0; i < nTest; i++) {
            sAE.reconstruct(testData[i], testOut[i]);
            for (int j = 0; j < nOutput; j++) {
                System.out.print(testOut[i][j] + " ");
            }
            System.out.println();
        }
    }
}
