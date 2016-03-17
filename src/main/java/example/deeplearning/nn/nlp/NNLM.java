package example.deeplearning.nn.nlp;

import example.deeplearning.nn.layers.HiddenLayer;
import example.deeplearning.nn.layers.LogisticRegression;
import example.deeplearning.nn.layers.ProjectionLayer;
import org.apache.commons.collections.BidiMap;
import org.apache.commons.collections.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Map;
import java.util.Random;
import java.util.function.IntToDoubleFunction;

import static example.deeplearning.nn.utils.*;



/**
 * Created by b1012059 on 2016/01/31.
 */
public class NNLM {
    //private static Logger log = LoggerFactory.getLogger(NNLM.class);
    private int nInput;
    private int nHidden;
    private int nOutput;
    private int vocab;
    private int dim;
    private int n;
    private double learningRate;
    private double decayRate;
    private ProjectionLayer pLayer;
    private HiddenLayer hLayer;
    private LogisticRegression logisticLayer;
    private Random rng;
    private IntToDoubleFunction learningType;

    public NNLM(int N, int vocab, int dim, int n, int nHidden, Random rng, double lr, double dr, String lrUpdateType){
        this.vocab = vocab;
        this.dim = dim;
        this.n = n;
        this.nInput = dim*2;
        this.nHidden = nHidden;
        this.nOutput = vocab;
        this.learningRate = lr;
        this.decayRate = dr;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        this.pLayer = new ProjectionLayer(N, vocab, dim, null, rng);
        this.hLayer = new HiddenLayer(this.nInput, nHidden, null, null, N, rng, "tanh");
        this.logisticLayer = new LogisticRegression(dim, nHidden, this.nOutput, N, rng, "tanh");

        if (lrUpdateType == "UpdateLR" || lrUpdateType == null) {
            this.learningType = (int epoch) -> updateLR(this.learningRate, this.decayRate, epoch);
        } else if(lrUpdateType == "AdaGrad") {
            //this.learningType = (int epoch) -> adaGrad(this.learningRate);
        } else if(lrUpdateType == "RMSProp"){
            //this.learningType = (int epoch) -> rmsProp(this.learningRate);
        } else {
            //log.info("Learning Update Type not supported!");
        }

    }

    public void train(Map<String, Integer> nGram, int epochs, NLP nlp) {
        double[][] lookUpInput = new double[n-1][dim];
        double[] hiddenInput;
        int[] teachInput = new int[vocab];
        double[] outLayerInput;
        double[][] dProjection;
        double[] dhOutput;
        double lr;
        int count = 1;

        /*
        N-gramの袋から着目単語の前n-1単語の分散表現を取り出す
        着目単語のときは教師データの作成
         */
        //log.info("Get LookUpTable and Create TeachData");
        for (Map.Entry<String, Integer> entry : nGram.entrySet()) {
            //log.info("Set " + count + "th N-gram");
            String[] words = entry.getKey().split(" ", 0);
            for (int i = 0; i < n; i++) {
                int vocabNumber = nlp.getWordToId().get(words[i]);
                if (i < n - 1) {
                    lookUpInput[i] = pLayer.lookUpTable(vocabNumber);
                    //log.info("LookUpTable " + vocabNumber + "th word");
                } else {
                    //log.info("TeachData:");
                    teachInput[vocabNumber] = 1;
                }
            }
            /*
            N-gram N回の学習
             */
            //log.info("Training N-gram");
            for (int epoch = 0; epoch < epochs; epoch++) {
                //初期化
                hiddenInput = ArrayUtils.addAll(lookUpInput[0], lookUpInput[1]);
                outLayerInput = new double[nHidden];
                dhOutput = new double[nHidden];
                dProjection = new double[n-1][nInput];
                lr = learningType.applyAsDouble(epoch);
                //log.info(String.valueOf(lr));
                System.out.println(String.valueOf(lr));

                hLayer.forwardCal(hiddenInput, outLayerInput);
                logisticLayer.train2(outLayerInput, lookUpInput, teachInput,
                        dProjection, dhOutput, lr);
                hLayer.backwardCal2(hiddenInput, outLayerInput, dProjection, dhOutput, lr);

                for(int j = 0; j < n - 1; j++) {
                    pLayer.backwardCal(nlp.getWordToId().get(words[j]), dProjection[j]);
                    //LookUpInputの更新
                    lookUpInput[j] = pLayer.lookUpTable(nlp.getWordToId().get(words[j]));
                }
            }
            count++;
        }
        //log.info("Finish N-gram");
    }

    /**
     * テストメソッド
     * @param nlp
     * @param word1
     * @param word2
     * @return
     */
    public void reconstruct(NLP nlp, String word1, String word2){
        double outLayerInput[] = new double[nHidden];
        double output[] = new double[vocab];
        double projection[][] = {pLayer.lookUpTable(nlp.getWordToId().get(word1)),
                pLayer.lookUpTable(nlp.getWordToId().get(word2))};
        BidiMap bidiMap = new DualHashBidiMap(nlp.getWordToId());

        hLayer.forwardCal(ArrayUtils.addAll(pLayer.lookUpTable(nlp.getWordToId().get(word1)),
                pLayer.lookUpTable(nlp.getWordToId().get(word2))), outLayerInput);
        logisticLayer.reconstruct2(outLayerInput, projection, output);

        int index = 0;
        for(int i = 0; i < output.length; i++) index = (output[index] >= output[i]) ? index : i;

        System.out.print((String) bidiMap.getKey(index));
    }

    /**
     * 単語同士のコサイン類似度を求める
     * @param nlp
     * @param word1
     * @param word2
     * @return
     */
    public double cosSim(NLP nlp, String word1, String word2){

        return cosineSimilarity(pLayer.lookUpTable(nlp.getWordToId().get(word1)),
                pLayer.lookUpTable(nlp.getWordToId().get(word2)));
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
        writeWordVectors(vocab, dim, nlp, pLayer.getwDI(), fileName);
    }
}
