package example.deeplearning.imageprocessing;

import java.awt.image.BufferedImage;
import java.io.File;

/**
 * Created by b1012059 on 2016/03/08.
 */
public class ImageProcessing {

    public ImageProcessing(){

    }

    public void imagePlot(){
        File file = new File("");
        BufferedImage write = new BufferedImage(100, 100, BufferedImage.TYPE_BYTE_GRAY);

    }

    /*
    public static void main(String[] args) throws IOException {
        File f = new File("test.jpg");
        BufferedImage read = ImageIO.read(f);
        int w = read.getWidth(), h = read.getHeight();
        BufferedImage write = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int c = read.getRGB(x, y);
                int r = 255 - r(c);
                int g = 255 - g(c);
                int b = 255 - b(c);
                int rgb = rgb(r, g, b);
                write.setRGB(x, y, rgb);
            }
        }

        File f2 = new File("ret.jpg");
        ImageIO.write(write, "jpg", f2);
    }
    */
}
