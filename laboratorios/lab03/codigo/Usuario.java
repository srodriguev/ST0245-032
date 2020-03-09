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

public class Usuario {

    public static void main(String[] args) throws IOException {

        System.out.println("Hola usuario");

        ejecucion();
    }

    public static void ejecucion() throws IOException {

        Scanner sc = new Scanner(System.in);
        Consultas c = new Consultas();
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        System.out.println("\n\n¿Qué consulta desea realizar");
        System.out.println("\n1. Consulta 1");
        System.out.println("2. Consulta 2");

        int a = sc.nextInt();

        switch (a) {

            case 1:

                System.out.println("Inserte la materia a consultar:");
                //String materia = sc.nextLine();
                String materia = reader.readLine();

                System.out.println();
                //

                System.out.println("Inserte el semestre a consultar:");
                //String semestre = sc.nextLine();
                String semestre = reader.readLine();

                c.c1(materia, semestre);
                break;

            case 2:
                System.out.println("Inserte el estudiante a consultar:");
                //String materia = sc.nextLine();
                String estudiante = reader.readLine();

                System.out.println();
                //

                System.out.println("Inserte el semestre a consultar:");
                //String semestre = sc.nextLine();
                String semestree = reader.readLine();
                
                c.c2(estudiante, semestree);
                break;

        }

        System.out.println("\n\n¿Desea realizar otra consulta");
        System.out.println("\n1. Si");
        System.out.println("2. No");

        int b = sc.nextInt();

        switch (b) {

            case 1:
                ejecucion();
                break;

            case 2:
                System.out.println("\n\nOk. Terminamos");
                break;

        }
    }

}
