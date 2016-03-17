package example.deeplearning.nn.nlp;

import example.deeplearning.nn.layers.LogisticRegression;
import example.deeplearning.nn.layers.RecurrentHLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Random;
import java.util.function.IntToDoubleFunction;

import static example.deeplearning.nn.utils.*;


/**
 * Created by b1012059 on 2016/02/15.
 */
public class RNNLM {
    private static Logger log = LoggerFactory.getLogger(RNNLM.class);
    private int nInput;
    private int nHidden;
    private int nOutput;
    private int vocab;
    private int dim;
    private double learningRate;
    private double decayRate;
    private RecurrentHLayer rLayer;
    private LogisticRegression logisticLayer;
    private Random rng;
    private IntToDoubleFunction learningType;

    public RNNLM(int N, int vocab, int dim, Random rng, double lr, double dr, String lrUpdateType){
        this.vocab = vocab;
        this.dim = dim;
        this.nInput = vocab;
        this.nHidden = dim;
        this.nOutput = vocab;
        this.learningRate = lr;
        this.decayRate = dr;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        this.rLayer = new RecurrentHLayer(vocab, nHidden, null, null, null, null, N, rng, "sigmoid");
        this.logisticLayer = new LogisticRegression(nHidden, this.nOutput, N, rng, "sigmoid");

        if (lrUpdateType == "UpdateLR" || lrUpdateType == null) {
            this.learningType = (int epoch) -> updateLR(this.learningRate, this.decayRate, epoch);
        } else if(lrUpdateType == "AdaGrad") {
            //this.learningType = (int epoch) -> adaGrad(this.learningRate);
        } else if(lrUpdateType == "RMSProp"){
            //this.learningType = (int epoch) -> rmsProp(this.learningRate);
        } else {
            log.info("Learning Update Type not supported!");
        }
    }

    public void train(Map<String, Integer> nGramm, int epochs, NLP nlp){
        double outLayerInput[];
        double rhInput[] = new double[nHidden];
        int[] teachInput = new int[vocab];
        int vocabNumber = 0;
        double lr;
        double dOutput[];

        log.info("Get LookUpTable and Create TeachData");
        for(int epoch = 0; epoch < epochs; epoch++) {
            lr = learningType.applyAsDouble(epoch);
            log.info("LearningRate: " + lr);
            for (Map.Entry<String, Integer> entry : nGramm.entrySet()) {
                String[] words = entry.getKey().split(" ", 0);
                for (int i = 0; i < words.length; i++) {
                    if (i < words.length - 1) {
                        vocabNumber = nlp.getWordToId().get(words[i]);
                        log.info("LookUpTable " + vocabNumber + "th word");
                    } else {
                        teachInput[nlp.getWordToId().get(words[i])] = 1;
                    }
                }

                outLayerInput = new double[nHidden];

                rLayer.forwardCal(vocabNumber, rhInput, outLayerInput);
                dOutput = logisticLayer.train(outLayerInput, teachInput, lr);
                rLayer.backwardCal(vocabNumber, null, outLayerInput, dOutput, logisticLayer.wIO, rhInput, lr);

                rhInput = outLayerInput;
            }
        }
    }

    /**
     * 単語同士のコサイン類似度を求める
     * @param nlp
     * @param word1
     * @param word2
     * @return
     */
    public double cosSim(NLP nlp, String word1, String word2){

        return cosineSimilarity(rLayer.lookUpTable(nlp.getWordToId().get(word1)),
                rLayer.lookUpTable(nlp.getWordToId().get(word2)));
    }

    /**
     * 車輪の再発明
     * DeepLearning4jのWordVectorSerializerクラスにある
     * writeWordVectorsを再実装
     * @param vocab
     * @param dim
     * @param nlp
     */
    public void writeWord(int vocab, int dim, NLP nlp, String fileName){
        writeWordVectors(vocab, dim, nlp, rLayer.getwIH(), fileName);
    }
}
