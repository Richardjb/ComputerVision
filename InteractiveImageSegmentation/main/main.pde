
import android.view.MotionEvent;
import ketai.camera.*;
import ketai.data.*;
import ketai.net.*;
import ketai.net.bluetooth.*;
import ketai.net.nfc.*;
import ketai.net.nfc.record.*;
import ketai.net.wifidirect.*;
import ketai.sensors.*;
import ketai.ui.*;

// for use of android sensors
import ketai.sensors.*;
// for android multitouch 
import android.content.Context.*;
import android.*;
import ketai.ui.*;
//import controlP5.*;

//import android.view.GestureDetector;

// sensor items
KetaiSensor sensor;
// Name of file
// base filename and database of images
String fileName = "input2.jpg";
String picDatabase[] = {"input1.jpg", "input2.jpg", "input3.jpg", "cosby.jpg", 
                          "familyGuyQuag.jpg", "tree1.jpg", "ant.jpg", "kidBug.jpg",
                            "worm.jpg", "ladybug.jpg",};
// dataType simular to MAT
PImage I; 
// handle button presses
//ControlP5 gui;
// where the segmentation process is performed
Segmentor segger = new Segmentor();
GraphOp graph;
KetaiGesture gesture;
float Size = 10;
float Angle = 0;
PImage img;
PImage backup; // used to restore image
PImage worker; // used for calc
String txt = "";
Boolean showInstructions = false;
Boolean showStartInstructions = true;
int picIndex = 2;
// Seeding data
Boolean didSegmentation = false;
Boolean choseForegrnd = false;
Boolean choseBackgrnd = false;
Boolean choseBoth = false;
int foreGrndx = -1, foreGrndy = -1;
int backGrndx = -1, backGrndy = -1;

float xScaleFactor;
float yScaleFactor;

void setup() 
{ 
  //fullScreen();
  // Set up default canvas size 
  // set to full screen size of the phone landscape mode
  orientation(LANDSCAPE);
  
  gesture = new KetaiGesture(this);

  I = loadImage(picDatabase[picIndex]);
  backup = loadImage(picDatabase[picIndex]);
  //backup = get();
  textSize(50);
  textAlign(CENTER);
  imageMode(CENTER);
  fill(color(230,230,0));
} 

public void ChangePicture(int theValue)
{
  //picIndex = (picIndex + 1) % 4;
  //fileName = picDatabase[picIndex];
  //reloadImage();
}

// Must be present for program to work  
void draw() 
{ 
  // sets background of app
  background(128);
  pushMatrix();
  translate(width/2, height/2);
  
  // removed rotation feature for user input calculation
  //rotate(Angle);
  
  //image(I, 0, 0, Size, Size);
  //image(I, 0, 0, Size + 500, Size + 500);
  // renders image to full screen size
  image(I, 0, 0, width, height);
  

  popMatrix();
  // draws circle around touch areas for selecting back and foreGround
  if (didSegmentation)
  {
    if (choseForegrnd)
    {
      ellipseMode(RADIUS);
      fill(200);
      ellipse((int)(foreGrndx/xScaleFactor), (int)(foreGrndy/yScaleFactor), 10, 10);
      
      if (choseBackgrnd)
      {
        choseBoth = true;
        if (!segger.RunningClustering)
        {
          fill(color(200,0,200));
          String txt = "Touch Screen to Rerun Algorithm";
          text(txt, width/4, height/4, width/4, height/4);
        }

        ellipseMode(RADIUS);
        fill(30);
        ellipse((int)(backGrndx/xScaleFactor), (int)(backGrndy/yScaleFactor), 10, 10);
      }
      else
      {
        fill(color(200,200,200));
        String txt = "Choose Background";
        text(txt, width/2, height/2, width/2, height/2);
      }
    }
    else
    {
      fill(color(100,100,100));
      String txt = "Choose foreground";
      text(txt, width/2, height/2, width/2, height/2);
    }
  }
  
  // shows instructions if screen is swiped
  if (segger.RunningClustering)
  {
    fill(color(200,23,33));
    String txt = "RUNNING SEGMENTATION";
    text(txt, width/8, height/8, width/2, height/2);
  }
} 

private int Index(int x, int y)
{
  return x + (y * I.width);
} 

private int Index(float x, float y)
{
  return (int)x + ((int)y * I.width);
} 

void onDoubleTap(float x, float y)
{
  // resets variable for new image
  choseBoth = false;
  choseBackgrnd = false;
  choseForegrnd = false;
  didSegmentation = false;
  
  picIndex = (picIndex + 1) % picDatabase.length;
  fileName = picDatabase[picIndex];
  background(200,200,200); 
  I = loadImage(fileName);
  backup = loadImage(fileName);
  image(I, 0,0, width, height);  
}


void onTap(float x, float y)
{
  // calculates scale factor to translate phones screen space
  // to the canvas' area
  xScaleFactor = (float)I.width / (float)width;
  yScaleFactor = (float)I.height/ (float)height;

  print ("Recalc xPos: " + x * xScaleFactor + " Recalc yPos: " + y * yScaleFactor);
  print("didseg: " + didSegmentation);
  if (choseBoth)
  {
    int foreInd = Index(foreGrndx, foreGrndy);
    int backInd = Index(backGrndx, backGrndy);
    backup.resize(150,0);
    PImage nwImage = segger.SeededDataAlg(foreInd, backInd, backup);
    I = nwImage;
    image(nwImage, 0, 0, width, height);
    
    choseBoth = false;
    choseBackgrnd = false;
    choseForegrnd = false;
    return;
  }
  
  if (didSegmentation)
  {
    print("choseFore: " + choseForegrnd);
    if (choseForegrnd)
    {
      print("choseBack: " + choseBackgrnd);
      if (choseBackgrnd)
      {
        SEED();
      }
      else
      {
        backGrndx = (int)(x * xScaleFactor);
        backGrndy = (int)(y * yScaleFactor);
        choseBackgrnd = true;
      }
    }
    if (!choseForegrnd)
    {
      choseForegrnd = true;
      foreGrndx = (int)(x * xScaleFactor);
      foreGrndy = (int)(y * yScaleFactor);
      
      print("backgroundx: " + backGrndx + " backgroundy: " + backGrndy);
    }
  }

  //print ("there was a tap, beginning graph prec");
  //graph = new GraphOp(I);
  //graph.printImage();
  //print ("The image is loaded in graph");
}

void RipImageForeGround()
{
  I.loadPixels();
  backup.loadPixels();
  PImage nwImage = createImage( I.width, I.height, RGB );
  nwImage.loadPixels();
  
  print("In Rip function");
  // shows original pixels but only for fore ground marked pixels
  for (int i = 0; i < I.width; i++)
  {
    for (int j = 0; j < I.height; j++)
    {
      if (I.pixels[Index(i,j)] == segger.clust1)
      {
        nwImage.pixels[Index(i,j)] = backup.pixels[Index(i,j)];
      } 
      else
      {
        nwImage.pixels[Index(i,j)] = segger.clust2;
      }
    }
  }
  print("Everything has been ripped");
  I = nwImage;
  I.loadPixels();
  image(nwImage, 0 , 0, width,height);
}

void SEED()
{
  print ("seed was called");
  print ("The foreground and background are, respectively");
  print (foreGrndx);
  print (backGrndx);
  I.loadPixels();
  // starting colors to be used with seeding algorithms
  color startingForeColor = I.pixels[Index(foreGrndx, foreGrndy)];
  color startingBackColor = I.pixels[Index(backGrndx, backGrndy)];
  
  // just expansion of region
  if (startingForeColor != segger.clust1)
  {
    ForeFloodFill(foreGrndx,foreGrndx);
    print("finished foreFlood");
  }
  
  BackFloodFill(backGrndx, backGrndx);
  print("finished backFlood");
  I = worker;
  I.loadPixels();
  // resets values for new run
  //didSegmentation = false;
  //choseForegrnd = false;
  //choseBackgrnd = false;
}

void ForeFloodFill(int x, int y)
{
  print("fore 1");
  int index = Index(x,y);
  I.loadPixels();
  worker.loadPixels();
  print("fore 2");
  
  //print ("img width: " + I.width + "img height: " + I.height + "product: " + (I.width * I.height));
  print ("index: " + index);
  // out of bounds base case
  if (index >= (I.width * I.height))
    return;
  print("fore 3");
  // reached background pixel
  if (I.pixels[index]  == segger.clust1)
    return;
  print("fore 4");
  worker.pixels[index] = segger.clust3;
  print("fore 5");
  ForeFloodFill(x+1,y);
  ForeFloodFill(x-1,y);
  ForeFloodFill(x,y+1);
  ForeFloodFill(x,y-1);
  print(" stopped one stack");
  worker.loadPixels();
}

void BackFloodFill(int x, int y)
{
  I.loadPixels();
  // if a pixel is not marked foreground then it is background
  for (int i = 0; i < (I.width * I.height); i++)
    if (worker.pixels[i] != segger.clust3)
      worker.pixels[i] = segger.clust1;
  return;
  /*
  int index = Index(x,y);
  I.loadPixels();
  // out of bounds base case
  if (index >= (I.width * I.height))
    return;
  // reached starter pixel base
  if (I.pixels[index] == segger.clust1)
    return;
  // recursive portion
  BackFloodFill(x + 1, y);
  */
}

void onLongPress(float x, float y)
{
  reloadImage(); 
  //backup = I;
  // resets values for new run
  didSegmentation = false;
  choseForegrnd = false;
  choseBackgrnd = false;
  choseBoth = false;
}

//the coordinates of the start of the gesture, 
//     end of gesture and velocity in pixels/sec
void onFlick( float x, float y, float px, float py, float v)
{
  final int right = 0, left = 1, up = 2, down = 3;
  
  // deadZone is ignore range to prevent false swypes
  // from being detected
  int deadZone = 50;
  int swipeType = -1;
  
  // swype assignment
  if (px > x && py < y + deadZone && py > y - deadZone)
  {
    swipeType = right;
  }
  if (px < x && py < y + deadZone && py > y - deadZone)
  {
    swipeType = left;
  }
  if (py > y && px < x + deadZone && px > x - deadZone)
  {
    swipeType = up;
  }
  if (py < y && py < x + deadZone && px > x - deadZone)
  {
    swipeType = down;
  }
  
  // getClusters(PImage, function, clusters)
  // events
  switch(swipeType)
  {
    case up:
          getClusters(I,0,2);
    break;
    case down:
          getClusters(I,0,2);
    break;
    case left:
          // removes background (green) and shows foreground
          if (didSegmentation)
          {
            choseForegrnd = false;
            choseBackgrnd = false;
            choseBoth = false;
            print("Calling ripping out foreground");
            RipImageForeGround();
          }
    break;
    case right:
          // removes background (green) and shows foreground
          if (didSegmentation)
          {
            choseForegrnd = false;
            choseBackgrnd = false;
            choseBoth = false;
            print("Calling ripping out foreground");
            RipImageForeGround();
          }
    break;
  }
}

void onPinch(float x, float y, float d)
{
  didSegmentation = false;
  // prevents movement if segmentation is done
  if (!didSegmentation)
    Size = constrain(Size+d, 10, 2000);
}

void onRotate(float x, float y, float ang)
{
  // prevents movement if segmentation is done
  if (!didSegmentation)
    Angle += ang;
}

// Keep these here
void mouseDragged()
{}
//void mousePressed()
//{}


public boolean surfaceTouchEvent(MotionEvent event) 
{
  //forward event to class for processing
  return gesture.surfaceTouchEvent(event);
}


void reloadImage() 
{ 
  background(200,200,200); 

  I = loadImage(fileName); 

  image(I, 0,0, width, height);  
} 

void getClusters(PImage img, int func, int clusters) 
{ 
  // prevents user from segmenting the image again until
  // the foreground and background calculations have been done
  if (didSegmentation)
    return;
    
  println("in get clusters");
  final int seg = 0, blur = 1; 
  PImage nwImage = createImage( I.width, I.height, RGB );
  img.resize(150,0);
  I.resize(150,0);
  
  worker = I;
  print("before got im");
  //PImage gotIm = get();
  //print("After got im");
  //gotIm.resize(150,0);
  switch (func)
  {
    case seg:
        nwImage = segger.Segment(/*img*/ img,clusters);
    break;
    case blur:
        nwImage = segger.Segment(/*img*/ img,clusters);
    break;
    default:
        nwImage = segger.Segment(/*img*/ img,clusters);
  }
  didSegmentation = true;
  
  print(nwImage);
  image(nwImage, 0 , 0, width,height);
} 