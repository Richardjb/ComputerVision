import java.util.Map;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

class Segmentor
{
  PVector[] clusters;
  HashMap<Integer,Integer>  clustHash; // 1d index -> cluster
  
  PVector[] centroids;
  int[] nwCentroids; // centroid[] using 1d array for mapping
  
  PImage buf;
  PImage clusteredImage;
  int K;
  
  // there will be a max of 4 clusters
  color clust1 = color(100,0,0);
  color clust2 = color(0,100,0);
  color clust3 = color(0,0,100);
  color clust4 = color(100,100,0);
  
  boolean RunningClustering = false;
  

  public Segmentor()
  {
  }
 
  public PImage Segment(PImage I, int numK)
  {
    buf = createImage( I.width, I.height, RGB ); 
    clusteredImage = createImage(I.width, I.height, RGB);
    buf = I;
    
    K = numK;
    // centroids = the mean of each cluster, there are k centroids
    clusters = new PVector[K];

    // Index in img -> cluster num
    clustHash = new HashMap<Integer, Integer>();
    nwCentroids = new int[K];
    // note the index of each cluster array matches the number
    // of clus+ters

    PerformAlg();
    RunningClustering = false;

    //return buf;
    return clusteredImage;
  }
  
  public PImage SeededDataAlg(int fore, int back, PImage imgReset)
  {
    //buf = createImage( imgReset.width, imgReset.height, ARGB ); 
    //buf = imgReset;
    
    
    // place user seeded data
    // assumes binary segmentation
    nwCentroids[0] = fore;
    nwCentroids[1] = back;
    
    for (int k = 0; k < 2; k++)
      {
       // maps each pixel to a cluster
       FillClusters();
       println("Filled Clusters to begin with");
       print ("The size of the matrix is: " + clustHash.size() + " The num pixels in pic: " + buf.pixels.length);
       
       for (int i = 0; i < nwCentroids.length; i++)
       {
         CalcNewCentroid(i);
       }
       println("calculated new centroids");
      }
     PlotClusters();
     println("finished plotting points");
     
     RunningClustering = false;
     
     return clusteredImage;
  }
  
  public void PerformAlg()
  {
     //initializes with random intensities
      PickRandomCentroids();
      println("Picked Starting Centriods");
      for (int k = 0; k < 2; k++)
      {
       // maps each pixel to a cluster
       FillClusters();
       println("Filled Clusters to begin with");
       print ("The size of the matrix is: " + clustHash.size() + " The num pixels in pic: " + buf.pixels.length);
       
       for (int i = 0; i < nwCentroids.length; i++)
       {
         CalcNewCentroid(i);
       }
       println("calculated new centroids");
      }
     PlotClusters();
     println("finished plotting points");
  }
  
  public void PlotClusters()
  {
    
    //buf.loadPixels();
    clusteredImage.loadPixels();
    
    /*
    File file = new File("//newfile.txt");
    String content = "This is the text content";
    
    try{ 
      FileOutputStream fop = new FileOutputStream(file);

      // if file doesn't exists, then create it
      if (!file.exists()) {
        file.createNewFile();
      }

      // get the content in bytes
      byte[] contentInBytes = content.getBytes();

      fop.write(contentInBytes);
      fop.flush();
      fop.close();

      System.out.println("Done");

    } catch (IOException e) {
      e.printStackTrace();
    }
    */
    
      
     // Using an enhanced loop to interate over each entry
      for (Map.Entry me : clustHash.entrySet()) 
      {
        println(me.getKey() + " is " + me.getValue());
          switch ((Integer)me.getValue())
          {
            case 0:
            clusteredImage.pixels[(Integer)me.getKey()] = clust1;
            break;
            case 1:
            clusteredImage.pixels[(Integer)me.getKey()] = clust2;
            break;
            case 2:
            clusteredImage.pixels[(Integer)me.getKey()] = clust2;
            break;
            case 3:
            clusteredImage.pixels[(Integer)me.getKey()] = clust3;
            break;
            case 4:
            clusteredImage.pixels[(Integer)me.getKey()] = clust4;
            break;
            default:
          }
      }
      clusteredImage.updatePixels();
  }
  
  public void FillClusters()
  {
    int clusterToBeAssignedTo = 0;
    float pointIntensity = 0, centroidStartIntensity = 0;
    float minDistance, compareValue;
    
    println("---------------------------------------------------Filling Clusters----------------------------------------");
    // readies the pixels
    buf.loadPixels();
    // assign all points in image to cluster
    for (int j = 0; j < buf.height; j++)
    {
      for (int i = 0; i < buf.width; i++)
      {
        int loc = Index(i,j);
        //print(loc);
        // gets point and centroid intensities
        pointIntensity = GetIntensity(loc);

        centroidStartIntensity = GetIntensity(nwCentroids[0]);
        clusterToBeAssignedTo = 0;
        
        minDistance = abs(pointIntensity - centroidStartIntensity);

        // compare with distance to all centroids
        
        for (int centroidIndex = 0; centroidIndex < nwCentroids.length; centroidIndex++)
        {
          compareValue =  abs(pointIntensity - GetIntensity(nwCentroids[centroidIndex]));
          
          if (compareValue < minDistance)
          {
            minDistance = compareValue;
            clusterToBeAssignedTo = centroidIndex;
          }
        }
        // prevents the same key from being registered twice with
        // different values
        if (clustHash.containsKey(loc))
            clustHash.remove(loc);
            
        clustHash.put(loc, clusterToBeAssignedTo);
      }
      print("running fill clusters");
      RunningClustering = true;
    }
    
    println("Done Filling Clusters");
    buf.updatePixels();
  }
  
  private float GetIntensity(int loc)
  {
    buf.loadPixels();
    
    float r = red (buf.pixels[loc]);
    float g = green (buf.pixels[loc]);
    float b = blue (buf.pixels[loc]);
    
    
    // calculates the grayscale by taking avg.
    float gray = (r + g + b)/3;
    buf.updatePixels();
    return gray;
  }
  
  public void CalcNewCentroid(int clusterIndex)
  {
    int sumOfIntensities = 0;
    int countOfElementsInCluster = 0;
    float startIntensity = 0;
    //finds the average intensity and sets as new centroid
     // Using an enhanced loop to interate over each entry
      for (Map.Entry me : clustHash.entrySet()) 
      {
        if ((Integer)me.getValue() == clusterIndex)
        {
          sumOfIntensities += GetIntensity((Integer)me.getKey());
          countOfElementsInCluster += 1;
          startIntensity =  GetIntensity((Integer)me.getKey());
        }
        //println(me.getKey() + " is " + me.getValue());
      }
      
      int avgIntensity = (Integer)(sumOfIntensities/countOfElementsInCluster);
      
      //new centroid will be the coord with the intensity nearest to the
      //avg. intensity, cluster index is the same as centroid index
      int nwCent = nwCentroids[clusterIndex];
      float smallestDist = sq(abs(avgIntensity - startIntensity));
      
      for (Map.Entry me : clustHash.entrySet()) 
      {
        if ((Integer)me.getValue() == clusterIndex)
        {
          if (sq(abs(avgIntensity - GetIntensity((Integer)me.getKey()))) < smallestDist)
          {
            nwCent = (Integer)me.getKey();
            smallestDist = sq(abs(avgIntensity - GetIntensity(nwCentroids[clusterIndex])));
          }
           
        }
        //println(me.getKey() + " is " + me.getValue());
      }
      nwCentroids[clusterIndex] = nwCent;
  }
  
  public void PickRandomCentroids()
  {
    //PVector temp = new PVector();
    int x,y;

    // pick random locations and set as centroids
    // the intensities will be used not distance
    for (int i = 0; i < K; i++)
    {
      
      x = int(random(0, buf.width));
      y = int(random(0, buf.width));
      
      // index yields intensity
      nwCentroids[i] = Index(x,y);
    }
    return;
  }

  private int Index(int x, int y)
  {
    return x + (y * buf.width);
  }

}