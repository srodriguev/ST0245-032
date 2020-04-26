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
public class NodeDirectory {

  public String usuario;
  public double tamaño;
  public String archivo;
  public ArrayList<NodeDirectory> elementos;

  public NodeDirectory(String usuario, double tamaño, String archivo){
    this.usuario=usuario;
    this.tamaño=tamaño;
    this.archivo=archivo;
  }
  @Override
  public String toString(){
    return "["+usuario+" "+tamaño+"] "+archivo;
  }

  
  
}