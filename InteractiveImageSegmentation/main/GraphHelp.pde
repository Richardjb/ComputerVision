import java.util.Map;

static class GraphHelp
{

  static int n=100000,i,j=0,k=0,p,q,x,ll,temp,y,answer;
  static int[] S = new int[n];
  static int[] T = new int[n];
  static int[] A = new int[n];
  static int[] O = new int[n];
  static int[] Tree = new int[n];
  static int[] Parent = new int[n];
  static ArrayList<Integer> P = new ArrayList<Integer>();
  
  public static int grow(int[][] graph, int[][] rgraph, int s, int t)
  {
    //grow S or T to find an augmenting path P from s to t
    n= graph.length;
    P.clear();
    for(p=0;p<n;p++)
    {
      if(A[p]==1)
      {
        //for each neighbour
        //System.out.println(p + "hey");
        for(q=0;q<n;q++)
        {
            if(((Tree[p]==1 && rgraph[p][q]>0) || (Tree[p]==2 && rgraph[q][p]>0)))
            {
              if(Tree[q]==0)
              {
                A[q]=1;
                Tree[q] = Tree[p];
                if(q==3)
                {
                //System.out.println(q + " hghjgj " + Tree[q] +" p is "+ p);
                }
                Parent[q] = p;
                              
              }
              else if(Tree[q]!=0 && Tree[q]!=Tree[p])
              {
                
                //P is path from s to t
                //if(p==1041)
                //System.out.println(p+"     "+q);
                if(Tree[p]==1)
                {
                  
                  x = p;
                  P.add(x);
                  while(x!=s)
                  {
                    if(p==967)
                    {
                      //System.out.println(x);
                    }
                    x = Parent[x];
                    P.add(x);
                    
                  }
                  
                  ll = P.size();
                  for(i=0;i<ll/2;i++)
                  {
                    temp = (int) P.get(i);
                    P.set(i, P.get(ll-i-1));
                    P.set(ll-i-1,temp);
                    
                  }
                  x = q;
                  P.add(x);
                  while(x!=t)
                  {
                    x = Parent[x];
                    P.add(x);            
                  }
                }
                else
                {
                  x = q;
                  P.add(x);
                  while(x!=s)
                  {
                    x = Parent[x];
                    P.add(x);
                    
                  }
                  ll = P.size();
                  for(i=0;i<ll/2;i++)
                  {
                    temp = (int) P.get(i);
                    P.set(i, P.get(ll-i-1));
                    P.set(ll-i-1,temp);  
                  }
                  x = p;
                  P.add(x);
                  while(x!=t)
                  {
                    x = Parent[x];
                    P.add(x);
                    
                  }
                  
                  
                }
                /*for(i=0;i<P.size();i++)
                {
                  System.out.print(P.get(i) + " ");
                  
                }*/
                P.add(-1);
                //System.out.println();
                return 0;
                
              }
              
            }
          
        }
        
        
      }
      A[p]=0;
    }
    
    return 0;
  }
  
  public static int valid_parent(int[][] graph, int[][] rgraph, int p,int s,int t)
  {
    int ans = -1;
    for(q=0;q<n;q++)
    {
        if(Tree[q]==Tree[p] && ((Tree[q]==1 && rgraph[q][p]>0) || (Tree[q]==2 && rgraph[p][q]>0)))
        {
          //is origin of q source or sink
          x=q;
          while(Parent[x]!=-1)
          {
            x = Parent[x];
          }
          if(x==s || x==t)
          {
            return q;
          }
        
        }
    
    }    
    return ans;
  }
  public static int maxflow(int[][] graph, int s, int t)
  {
    n= graph.length;
    answer=0;
    //boolean[] visited = new boolean[n];
    int[][] rgraph = new int[n][n];
    
    for(i=0;i<n;i++)
    {
      for(j=0;j<n;j++)
      {
        rgraph[i][j] = graph[i][j];
      }  
    }
    //for(i=0;i<20;i++)
    {
      //System.out.println(rgraph[i][1041]);
    }
    for(i=0;i<n;i++)
    {
      S[i] = T[i] = A[i] = O[i] = Tree[i] = 0;
      Parent[i] = -1;
    }
    A[s] = S[s] = 1;
    A[t] = T[t]=1;
    //in Tree: 1 is set S, 2 is set T and 0 is neither
    Tree[s] = 1;
    Tree[t] = 2;
  
    while(true)
    {
      //growth stage
      grow(graph,rgraph,s,t);
      
      //  if P =  terminate
      if(P.isEmpty())
      {
        break;
        
      }
      
      //***************augment on P*******************//
      
      //find bottleneck
      x = s;
      i=1;
      y = (int)P.get(i);
      int min_flow = rgraph[x][y];
      while(x!=t)
      {
        if(rgraph[x][y] < min_flow)
        {
          min_flow = rgraph[x][y];
        }
        i++;
        x=y;
        y = (int)P.get(i);
      }
      
      
      //update residual graph
      x = s;
      i=1;
      y = (int)P.get(i);
      answer+= min_flow;
      
      while(x!=t)
      {
        rgraph[x][y] -= min_flow;
        rgraph[y][x] += min_flow;
        
        //if edge is saturated
        if(rgraph[x][y]==0)
        {
          if(Tree[x]==1 && Tree[y]==1)
          {
            Parent[y] = -1;
            O[y] = 1;
          }
          if(Tree[x]==2 && Tree[y]==2)
          {
            Parent[x] = -1;
            O[x] = 1;
          }
        }
        i++;
        x=y;
        y = (int)P.get(i);
      }
      
      //System.out.println("the min flow is " + min_flow);
      /*for(i=0;i<n;i++)
      {
        for(j=0;j<n;j++)
        {
          System.out.print(rgraph[i][j]+ "  ");
        }  
        System.out.println();
      }
      System.out.println();*/
      //*********adopt orphans***************//
      int check=1;
      while(check>0)
      {
        check=0;
      
      for(p=0;p<n;p++)
      {
        //for each orphan node
        if(O[p]==1)
        {
          //remove from list of orphans
          O[p]=0;
          
          //process
          x = valid_parent(graph,rgraph,p,s,t);
          
          //x is a valid parent
          if(x>=0)
          {
            Parent[p] = x;
          }
          //no valid parent found
          else
          {
            for(q=0;q<n;q++)
            {
              if(Tree[q]==Tree[p])
              {
                if((Tree[q]==1 && rgraph[q][p]>0) || (Tree[q]==2 && rgraph[p][q]>0))
                {
                  A[q] = 1;
                }
                
                if(Parent[q]==p)
                {
                  O[q]=1;
                  Parent[q]=-1;
                }
              }
            }
            Tree[p]=0;
            A[p]=0;
          }
        }
      }
      for(p=0;p<n;p++)
      {
        if(O[p]==1)
        {
          check=1;
        }
      }
      }
    }
    return 0;
  }
}