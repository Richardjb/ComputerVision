

class GraphOp
{
  //Mat image;
  //Mat procImg; NOT USED
  //Mat Backup;
  
  PImage image;
  PImage Backup;
  
  int [][] arr; // matrix with 0 ,1,2
  int [][] fgraph; // n pixel matrix
  int [] intensity; // black  white image
  int [] usermark;
  int objcount=0;
  int backcount=0;
  int col=0;
  int row =0;
 public GraphOp(PImage img)
 {
   //image = Highgui.imread(s); read in image
   image = img;
   Backup= image; // removes referrence
   PImage mat1 = createImage(image.width, image.height, RGB); // new buffer
   //Mat mat1 = new Mat(image.height(),image.width(),CvType.CV_8UC1);
   //Imgproc.cvtColor(image, mat1, Imgproc.COLOR_RGB2GRAY);
   image=mat1; // set image
   //col=image.cols();
   //row=image.rows();
   col = image.width;
   row = image.height;
   arr=new int[col][row];
   intensity =new int[col*row];
   usermark = new int[col*row];
   print("here 0");
   int sze = (col*row) + 2;
   print("sze: " + sze);
   fgraph = new int[sze][sze];
   print("here 1");
   
   print("here 2");
   //procImg=new Mat(col, row, Highgui.IMREAD_ANYCOLOR);
   
 }
 public void setPoint(int x,int y,int val)
 {
   arr[x][y]=val;
 }
 public int getCol()
 {
   return col;
 }
 public int getRow()
 {
   return row;
 }
 
 public void bgcalc()
 {
   image=Backup;
   //Imgproc.Canny(image, image, 300, 600, 5, true); // resets image
   //Highgui.imwrite("test.jpg", image);
 }
 
 public void bgsub()
 { 
   //BackgroundSubtractorMOG sub = new BackgroundSubtractorMOG(3, 4, 0.8); // background removal algorithm
   //Mat mGray = new Mat();
   Mat mRgb = new Mat();
   Mat mFGMask = new Mat();

   
     // I chose the gray frame because it should require less resources to process
     // Imgproc.cvtColor(image, mRgb, Imgproc.COLOR_GRAY2RGB); //the apply function will throw the above error if you don't feed it an RGB image
     // sub.apply(mRgb, mFGMask); //apply() exports a gray image by definition //
       
    //BackgroundSubtractorMOG bs = new BackgroundSubtractorMOG();
    //Mat mat1 = new Mat(image.height(),image.width(),CvType.CV_8UC1);
    //Imgproc.cvtColor(image, mat1, Imgproc.COLOR_RGB2GRAY);
    //image=mat1;
    //bs.apply(image, procImg);

    //Highgui.imwrite("test.jpg", mFGMask); // show image
    
    //bs.apply(image, procImg);
//    Highgui.imwrite("test.jpg", procImg);
 }
 
 
public void printImage()
{
  print("in print image 1");
  image=Backup;
  print("in print image 1.5");
  for(int i=0;i<col;i++)
  {
    for(int j=0;j<row;j++)
    {
      print("in print image 1.8");
      int sum=0;
      //float [] ab = image.get(j, i);
      //float [] ab = image.get(j, i);
       color ab = get(i, j, image); // returns color associated with index
       print("in print image 1.9");
       sum += (int)red(ab);
       sum += (int)blue(ab) * 1000;
       sum += (int)green(ab) * 1000000;
        //sum=sum+(int)ab[0];
        //sum=sum+(int)ab[1]*1000;
        //sum=sum+(int)ab[2]*1000000;
      
      intensity[i*row+j]=sum;
      //intensity[i*row+j]=(int) image.get(j, i)[0];
    }
  }
  print("in print image 2");
  for(int i=0;i<col;i++)
  {
    for(int j=0;j<row;j++)
    {
      usermark[i*row+j]=arr[i][j];
    }
  }
  
  print("in print image 3");
  objcount=0;
  backcount=0;
  for(int i=0;i<col;i++)
  {
    for(int j=0;j<row;j++)
    {
      if(arr[i][j]==1)
      {//System.out.print("col: "+i+"row: "+j+""+arr[i][j]+"\n");
        objcount++;
      }
      if(arr[i][j]==2)
      {//System.out.print("col: "+i+"row: "+j+""+arr[i][j]+"\n");
        backcount++;
      }
    }
  }
  
  print("in print image 4");
  int n=col*row;
  int k=10,lam=7;
  int[][] r=new int [n][2];
  int [][] b=new int[n][n];
  int max=0,x=0;
  //int cost=1;
  int sigma=2;
  print("in print image 5");
  for(int i=0;i<n;i++)
  { 
  r[i][0]=(int) (Math.log(intensity[i]/objcount));
  r[i][1]=(int) (Math.log(intensity[i]/backcount));
  }
  print("in print image 6");
  for(int i=0;i<n;i++)
  {   
    x=0;
    for(int j=0;j<n;j++)
    {
      if((j==(i-1)) || (j==(i+1)) || j==(i-row) || j==(i+row) )
      {
        b[i][j]=(int) ( Math.pow(Math.E, -1*Math.pow((intensity[i]-intensity[j])/(2*sigma*sigma), 2)));
        x+=b[i][j];
      }
      if( j==(i+row-1) || j==(i+row+1) || j==(i-row-1) || j==(i-row+1))
      {
        b[i][j]=(int) (Math.pow(Math.E, -1*Math.pow((intensity[i]-intensity[j])/(2*sigma*sigma), 2))*1.44);
        x+=b[i][j];
      }
    }
    if(x>max)
    {
      max=x;
    }
  }
  print("in print image 7");
  k=max+1;
/*  Random rand=new Random();
  for(int i=0;i<n;i=i+10)
  {
    int o=rand.nextInt(2);
    usermark[i]=o+1;
  }*/
  for(int i=0;i<n;i++)
  {
    if(usermark[i]==2)
    {
      fgraph[i][n+1]=k;
      fgraph[n+1][i]=k;

    }
    else if(usermark[i]==1)
    {
      fgraph[i][n+1]=0;
      fgraph[n+1][i]=0;

    }
    else
    {
      fgraph[i][n+1]=lam*r[i][0];
      fgraph[n+1][i]=lam*r[i][0];
    }
  }
  print("in print image 8");
  for(int i=0;i<n;i++)
  {
    if(usermark[i]==2)
    {
      fgraph[i][n]=0;
      fgraph[n][i]=0;
    }
    else if(usermark[i]==1)
    {
      fgraph[i][n]=k;
      fgraph[n][i]=k;
    }
    else
    {
      fgraph[i][n]=lam*r[i][1];
      fgraph[n][i]=lam*r[i][1];
    }
  }
  print("in print image 9");
  for(int i=0;i<n;i++)
  {
    for(int j=0;j<n;j++)
    {
      if((j==(i-1)) || (j==(i+1)) || j==(i-row) || j==(i+row) || j==(i+row-1) || j==(i+row+1) || j==(i-row-1) || j==(i-row+1))
      {
      fgraph[i][j]=b[i][j];
      }
    }
  }
  //Mat fimage=

  //ff.ford_fulkerson(fgraph, n, n+1);
   GraphHelp.maxflow(fgraph,n, n+1);
   System.out.println("The main flow is " + GraphHelp.answer); 
   
   
   float[] data1=new float[3];
   data1[0]=0;
   data1[0]=0;
   data1[0]=0;
   
   float[] data2=new float[3];
   data2[0] = 255;
   data2[0] = 255;
   data2[0] = 255;
   image = Backup;
   print("in print image 10");
   for(int i=0;i<col;i++)
    {
      for(int j=0;j<row;j++)
      {
        
        //if(Main.Tree[i*row+j]==1)
        //{
        //procImg.put(j, i, data1);
        //}
        int temp=i*row+j;
         if(GraphHelp.Tree[i*row+j]==2)
        {
        //image.put(j, i, data1);
        image = put(i, j, data1[0], image);
        }
         if(temp>col){
         if(GraphHelp.Tree[temp-1]==2 && GraphHelp.Tree[temp+1]==2 && GraphHelp.Tree[temp-col]==2)
         {
           image = put(i, j, data1[0], image);
            //image.put(j, i, data1);
         }
         }
         if(temp<(col-2)*(row-2) && temp >0){
           if(GraphHelp.Tree[temp-1]==2 && GraphHelp.Tree[temp+1]==2 && GraphHelp.Tree[temp+col]==2)
           {
              //image.put(j, i, data1);
              image = put(i, j, data1[0], image);
           }
           }
        
      }
      
    }
   image=Backup;
  // Imgproc.Canny(image, image, 300, 600, 5, true); 
  // Highgui.imwrite("test.jpg", image); // display image
  // System.out.println("The ford flow is " + ff.ford_fulkerson(fgraph,n, n+1));
  /* System.out.println("the first set is");
     for(int i=0;i<ff.j;i++)
     {
       System.out.print(ff.set1[i]+ "  ");
     }
     System.out.println();
     System.out.println("The second set is");
     for(int i=0;i<ff.k;i++)
     {
       System.out.print(ff.set2[i]+ "  ");
     }
*/
}
  public color get(int x, int y, PImage I)
  {
    //return GetIntensity(Index(x,y), I);
    int index = Index(x,y);
    I.loadPixels();
    return I.pixels[index];
  }
  
  private int Index(int x, int y)
  {
    return x + (y * img.width);
  }
  
  public PImage put(int x, int y, float value, PImage I)
  {
    I.loadPixels();
    
    int index = Index(x,y);
    I.pixels[index] = color(value, value, value);
    
    I.updatePixels();
    
    return I;
  }
  
  private float GetIntensity(int loc, PImage I)
  {
    I.loadPixels();
    
    float r = red (I.pixels[loc]);
    float g = green (I.pixels[loc]);
    float b = blue (I.pixels[loc]);
    
    // calculates the grayscale by taking avg.
    float gray = (r + g + b)/3;
    // print("r: " + r + "g: " + g + "b: " + b + "gray: " + gray + "\tregular intensity: " + buf.pixels[loc]);
    return gray;
  }

}