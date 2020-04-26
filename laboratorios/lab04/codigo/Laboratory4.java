/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author isabellaqv
 */
import java.util.ArrayList;

public class Laboratory4 
{

    public static NodeDirectory root;
    
    public Laboratory4(NodeDirectory r)
    {
        Laboratory4.root = r;
    }
    
    public void addFile(NodeDirectory d, NodeDirectory n){

        if(d == null) d=n;
        else addFileAux(d, n);

    }
    private void addFileAux(NodeDirectory n, NodeDirectory toAdd){

       n.elementos.add(toAdd);

    }
    
    public void search(String name){

       if(searchAux(root, name) == null ) System.out.println("No such file or directory");

       else System.out.println(searchAux(root, name).archivo);

    }
    public NodeDirectory searchAux(NodeDirectory n, String name){
      
      if (n!=null){
        if(n.archivo.equals(name)) return n;
        else if (n.elementos.isEmpty()==false){
          for(NodeDirectory ele : n.elementos){
            searchAux(ele, name);
          }
        }
      }
      return null;
    }
    
    public ArrayList<String> searchByAuthor(NodeDirectory n, String a){
      ArrayList<String> res = new ArrayList<>();

      if(n != null){
        if(n.usuario.equals(a)){ 
          res.add(n.archivo);
          System.out.println(n.toString());}
        else if (n.elementos.isEmpty()==false){
        for(NodeDirectory ele : n.elementos){
            searchByAuthor(ele, a);
          }
        }
      }

      return res;
    }
    
    public ArrayList<String> searchBySize(NodeDirectory n, double s){
        ArrayList<String> res = new ArrayList<>();

      if(n != null){
        if(n.tamaño >= s) {
          res.add(n.archivo);
          System.out.println(n.toString());}
        else if (n.elementos.isEmpty()==false){
        for(NodeDirectory ele : n.elementos){
            searchBySize(ele, s);
          }
        }
      }

      return res;
    }
    
    public void printFiles(NodeDirectory a){
      System.out.println(a.toString());
      if (a.elementos.isEmpty()==false){
        for(NodeDirectory ele : a.elementos){
            System.out.println(ele.toString());
            }
      }
    }
}