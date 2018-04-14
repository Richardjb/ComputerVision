

class Mat
{
  PImage img;
  
  Mat()
  {
  }
  
  public float get(int x, int y)
  {
    return GetIntensity(Index(x,y));
  }
  
  private int Index(int x, int y)
  {
    return x + (y * img.width);
  }
  
  public void put(int y, int x, int value)
  {
    img.loadPixels();
    
    int index = Index(x,y);
    img.pixels[index] = color(value, value, value);
  }
  private float GetIntensity(int loc)
  {
    img.loadPixels();
    
    float r = red (img.pixels[loc]);
    float g = green (img.pixels[loc]);
    float b = blue (img.pixels[loc]);
    
    // calculates the grayscale by taking avg.
    float gray = (r + g + b)/3;
    //print("r: " + r + "g: " + g + "b: " + b + "gray: " + gray + "\tregular intensity: " + buf.pixels[loc]);
    return gray;
  }
  
  
  
  
}