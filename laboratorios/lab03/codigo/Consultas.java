/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package laboratorio3;
import java.util.LinkedList;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.*;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Esta clase permite realizar
 * estadisticas a cada tipo de dato
 */ 
public class Consultas extends Usuario {

    
    /**
     * Este metodo permite realizar la estadistica promedio
     */
    public void c1(String materia, String semestree){
        
       LinkedList datos = MarcoDeDatos.getMyData();
       
       Dato current = (Dato) datos.get(0);
       //System.out.println(current.nombre);
       int range = datos.size();
       for (int i=0;i<range;i++)
       {
           if ((current.codMateria).equals(materia) && (current.semestre).equals(semestree))
           {
               System.out.println("Estudiante "+(i+1)+", "+current.nombre+", cumple con los requisitos");
           }          
           
       }
       
    }

    public void c2(String estudiantee, String semestree){
        
        LinkedList datos = MarcoDeDatos.getMyData();
        
        Dato current = (Dato) datos.get(0);
       //System.out.println(current.nombre);
       int range = datos.size();
       for (int i=0;i<range;i++)
       {
           if ((current.codMateria).equals(estudiantee) && (current.semestre).equals(semestree))
           {
               System.out.println("Estudiante "+(i+1)+", "+current.nombre+", cumple con los requisitos");
           }          
           
       }
       
    }
}