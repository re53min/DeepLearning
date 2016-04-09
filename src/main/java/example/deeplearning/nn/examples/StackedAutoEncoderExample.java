package example.deeplearning.nn.examples;

import example.deeplearning.nn.multilayer.StackedAutoEncoder;

/**
 * Created by b1012059 on 2016/03/12.
 */
public class StackedAutoEncoderExample {

    public static void main(String args[]){

        //入力データ
        double inputData[][] = {
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
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
        };

        StackedAutoEncoder sAE = new StackedAutoEncoder.Builder()
                .inputSize(inputData.length)
                .nIn(10)
                .nHidden(new int[]{8, 6, 4})
                .nOut(2)
                .activation("ReLU")
                .numEpochs(1000)
                .useAdaGrad(false)
                .seed(123)
                .dropOut(0.5)
                .build();

        //pre-training
        sAE.preTraining(inputData);
        //fine-tuning
        sAE.fineTuning(inputData, teachData);

        //test
        int nTest = 2;
        double testOut[][] = new double[nTest][sAE.getNOut()];

        //Output
        for(int i = 0; i < nTest; i++) {
            sAE.reconstruct(testData[i], testOut[i]);
            for (int j = 0; j < sAE.getNOut(); j++) {
                System.out.print(testOut[i][j] + " ");
            }
            System.out.println();
        }
    }
}
