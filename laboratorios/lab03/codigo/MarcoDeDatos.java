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
import java.util.ArrayList;
import java.util.List;

/**
 * Esta clase permite organizar y disttibuir correctamente los datos para crear
 * un ????? Array/ArrayList /SingleChainedList donde cada posición tiene una
 * variable de tipo Dato
 */
public class MarcoDeDatos {

    /**
     * ????? donde se almacenan los datos leidos *
     */
    private static LinkedList<Dato> miLista = new LinkedList<>();
    
    public static LinkedList getMyData()
    {
        return miLista;
    }

    /**
     * Este método recibe el nombre del archivo que se va a leer. Se trata de
     * abrir el archivo, si se puede abrir, se leen y se muestran impresos los
     * datos y se retornan en un ????? que almacena referencias a objetos tipo
     * Dato.
     *
     * @return 
     * @throws java.io.IOException
     */
    public LinkedList leerDatos(String nombre) throws IOException {
        //List<Dato> al = new LinkedList<Dato> ();
        File file = new File(nombre);
        //BufferedReader br = new BufferedReader(new FileReader(file));
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(
                nombre), "UTF-16"));

        String st;
        int cont = 0;

        while ((st = br.readLine()) != null) {
            if (cont == 0) {
                //Linea con títulos
                
            } else if (cont % 2 != 0) {
                //linea intermedia
                
            } else {

                //String nombre_, int codigo, String codMateria, String semestre, int grupo, String evaluacion, double porcentaje, String descripcion, String materia, int nota, int definitiva
                // 0 nomb / 1 cod / 2 codMat/ 3 sem / 4 grupo / 5 eval / 6 porc / 7 descrip / 8 mat / 9 nota / 10 def
                String[] stVec = st.split(",");

                //System.out.println("De: " + cont + ",");
                for (int i = 0; i < stVec.length; i++) {
                    //System.out.println("ind: " + i + ", es: " + stVec[i] + "| ");
                }

                int codigo = 0, grupo = 0, nota = 0, definit = 0;
                double porcentaje = 0;
                boolean success = false;

                try {
                    if (stVec[1] == null || stVec[1].equals("") || stVec[1].equals("NULL")) {
                        codigo = 0;
                    } else {
                        codigo = Integer.parseInt(stVec[1].trim());
                    }

                    if (stVec[4] == null || stVec[4].equals("") || stVec[4].equals("NULL")) {
                        grupo = 0;
                    } else {
                        grupo = Integer.parseInt(stVec[4].trim());
                    }

                    if (stVec[8] == null || stVec[8].equals("") || stVec[8].equals("NULL")) {
                        porcentaje = 0;
                    } else {
                        porcentaje = Double.parseDouble(stVec[8].trim());
                    }

                    if (stVec[12] == null || stVec[12].equals("") || stVec[12].equals("NULL")) {
                        nota = 0;
                    } else {
                        nota = Integer.parseInt(stVec[12].trim());
                    }

                    if (stVec[13] == null || stVec[13].equals("") || stVec[13].equals("NULL")) {
                        definit = 0;
                    } else {
                        definit = Integer.parseInt(stVec[13].trim());
                    }
                    success = true;
                } catch (Exception e) {
                    success = false;
                }

                if (stVec[0] == null) {
                    stVec[0] = "N/A";
                }
                if (stVec[2] == null) {
                    stVec[2] = "N/A";
                }
                if (stVec[3] == null) {
                    stVec[3] = "N/A";
                }
                if (stVec[7] == null) {
                    stVec[7] = "N/A";
                }
                if (stVec[9] == null) {
                    stVec[9] = "N/A";
                }
                if (stVec[10] == null) {
                    stVec[10] = "N/A";
                }

                if (success) {
                    Dato dato1 = new Dato(stVec[0], codigo, stVec[2], stVec[3], grupo, stVec[5], porcentaje, stVec[7], stVec[8], nota, definit);
                    miLista.add(dato1);
                }

            }
            cont++;
        }
        System.out.println("num lineas: " + cont);
        return miLista;
    }

    /**
     * Programa principal. Se leen los datos almacenados en el archivo
     * "datos.txt"
     *
     * @param args
     *
     */
    public static void main(String[] args) {
        MarcoDeDatos ldd = new MarcoDeDatos();
        System.out.println("Voy a leer los datos");
        try {
            ldd.leerDatos("NOTAS ST0242.csv");
            ldd.leerDatos("NOTAS ST0245.csv");
            ldd.leerDatos("NOTAS ST0247.csv");
        } catch (IOException ex) {
            System.out.println("mala leida");
        }

        System.out.println("Ya leí los datos");

    }
}
