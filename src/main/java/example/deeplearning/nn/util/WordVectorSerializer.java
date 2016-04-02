package example.deeplearning.nn.util;

import example.deeplearning.nn.nlp.NLP;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by b1012059 on 2016/03/31.
 */
public class WordVectorSerializer {

    public WordVectorSerializer(){
    }

    /**
     * 車輪の再発明
     * DeepLearning4jのWordVectorSerializerクラスにある
     * writeWordVectorsを再実装
     * @param vocab
     * @param dim
     * @param nlp
     */
    public static void writeWordVectors(int vocab, int dim, NLP nlp, double w[][], String fileName){
        try {
            BufferedWriter write = new BufferedWriter(new FileWriter(new File(fileName + ".txt"), false));
            boolean flag = true;
            StringBuilder sb = new StringBuilder();
            sb.append(vocab);
            sb.append(" ");
            sb.append(dim);
            sb.append("\n");
            write.write(sb.toString());

            for (String key : nlp.getWordToId().keySet()) {
                sb = new StringBuilder();
                sb.append(key);
                sb.append(" ");

                for(int i = 0; i < dim; i++){
                    sb.append(w[i][nlp.getWordToId().get(key)]);
                    if(i < dim - 1) sb.append(" ");
                }
                sb.append("\n");
                write.write(sb.toString());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
