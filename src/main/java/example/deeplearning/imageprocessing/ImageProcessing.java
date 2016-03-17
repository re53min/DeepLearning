package example.deeplearning.imageprocessing;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * Created by b1012059 on 2016/03/08.
 */
public class ImageProcessing {

    public ImageProcessing(){

    }

    public void imagePlot(double px[], int height, int width) {
        //MemoryImageSource producer = new MemoryImageSource(width, height, px, 0, width);
        Image img = null;//createImage(producer);

        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = bi.createGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();

        //PNG画像ファイルとして保存
        try {
            ImageIO.write(
                    bi, "png", java.io.File.createTempFile("test", ".png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
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
