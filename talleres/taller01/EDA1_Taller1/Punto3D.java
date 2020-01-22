
/**
 * Write a description of class Punto3D here.
 * 
 * @author (your name) 
 * @version (a version number or a date)
 */
public class Punto3D
{
    // instance variables - replace the example below with your own
    private double x, y, z;
    

    /**
     * Constructor for objects of class Punto3D
     */
    public Punto3D(double x, double y, double z)
    {
        this.x=x;
        this.y=y;
        this.z=z;
    }

    public double x() 
    {
        return x;
    }
    
    public double y() 
    {
        return y;
    }
    
    public double z()
    {
        return z;
    }
    
    public double distanciaPuntos(Punto3D otro)
    {
        return Math.sqrt( (otro.x()-this.x) + (otro.y()-this.y) + (otro.z-this.z) );
    }
}
