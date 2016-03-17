package example.deeplearning.nn.examples;

import example.deeplearning.nn.layers.LogisticRegression;

import java.util.Random;

/**
 * Created by b1012059 on 2016/03/12.
 */
public class LogisticRegressionExample {

    public static void main(String args[]){

        int nInput = 6;
        int nOutput = 2;
        int nTest = 2;
        int epochs = 500;
        double learningRate = 0.1;
        Random rng = new Random(123);

        double inputData[][] = {
                {1., 1., 1., 0., 0., 0.},
                {1., 0., 1., 0., 0., 0.},
                {1., 1., 1., 0., 0., 0.},
                {0., 0., 1., 1., 1., 0.},
                {0., 0., 1., 1., 0., 0.},
                {0., 0., 1., 1., 1., 0.}
        };

        int teachData[][] = {
                {1, 0},
                {1, 0},
                {1, 0},
                {0, 1},
                {0, 1},
                {0, 1}
        };

        double testData[][]= {

                {0., 1., 0., 0., 0., 1.},
                //{0., 0., 1., 1., 1., 0.}
        };

        double testOutput[][] = new double[nTest][nOutput];

        LogisticRegression logReg = new LogisticRegression(nInput, nOutput, inputData.length, rng);

        for(int epoch = 0; epoch < epochs; epoch++){
            for(int i = 0; i < inputData.length; i++){
                logReg.train(inputData[i], teachData[i], learningRate);
                //if(learningRate > 1e-5) learningRate *= 0.995;
                //log.info(String.valueOf(learningRate));
            }
        }


        System.out.println("-----------------TEST-----------------");
        for(int i = 0; i < nTest; i++){
            logReg.reconstruct(testData[i], testOutput[i]);
            for(int j = 0; j < nOutput; j++){
                System.out.print(testOutput[i][j] + " ");
            }
            System.out.println();
        }
    }
}
