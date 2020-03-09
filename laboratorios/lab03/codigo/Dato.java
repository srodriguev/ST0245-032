/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package laboratorio3;
public class Dato 
{

    String nombre;
    int codigo;
    String codMateria;
    String semestre;
    int grupo;  
    String evaluacion;
    double porcentaje;
    String descripcion;
    String materia;
    int nota;
    int definitiva;

    public Dato(String nombre, int codigo, String codMateria, String semestre, int grupo, String evaluacion, double porcentaje, String descripcion, String materia, int nota, int definitiva) {

        this.nombre = nombre;
        this.codigo = codigo;
        this.codMateria = codMateria;
        this.semestre = semestre;
        this.grupo = grupo;
        this.evaluacion = evaluacion;
        this.porcentaje = porcentaje;
        this.descripcion = descripcion;
        this.materia = materia;
        this.nota = nota;
        this.definitiva = definitiva;
    }

}