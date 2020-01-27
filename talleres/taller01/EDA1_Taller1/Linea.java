
/**
 * linea 2D
 *
 * Stiven yepes - sara Rodriguez
 * 2020
 */
public class Linea
{
    
    private int x1, x2, y1, y2;
    private Punto[] puntos;

    
    public Linea(int x1, int x2, int y1, int y2)
    {
        this.x1=x1;
        this.x2=x2;
        this.y1=y1;
        this.y2=y2;
        puntos = new Punto[((x2-x1)+(y2/y1))/2];
    }
    
   
    public Punto[] PuntosIntermedios(int x1, int x2, int y1, int y2) // algoritmo de Bresenham 
    {
    
        int dx = x2-x1;  // distancia que se desplazan 
        int dy = y2-y1;
        
        int IncYi, IncXi; // incremento en Y X inclinado 
        
        if(dy >= 0) 
            IncYi =1;
        else{
            dy = -dy;
            IncYi = -1;
        }
        
        if(dx>=0) IncXi = 1;
        else{
            dx = -dx;
            IncXi = -1;
        }
        
        int IncYr, IncXr; // incremento en Y X recto
        
        if(dx >= dy){
        IncYr = 0;
        IncXr = IncXi;
        }
        else{
        IncXr = 0;
        IncYr = IncYi;
        }
        
        double avr = 2*dy; // avance recto 
        double av = avr-dx; // avance
        double avi = av-dx; // avance inclinado 
        
        int x=x1;
        int y=y1;
        puntos[0]= new Punto(x,y);
        for(int i=1; x<=x2; i++){
            if(av>=0){
              x = x+IncXi;     // X aumenta en inclinado.
              y = y+IncYi;     // Y aumenta en inclinado.
              av = av+avi;     // Avance Inclinado
            }
            else{
              x = x+IncXr;     // X aumenta en recto.
              y = y+IncYr;    // Y aumenta en recto.
              av = av+avr;    // Avance Recto
            }
            puntos[i] = new Punto(x,y);
        }
            // referencia algoritmo: https://es.wikipedia.org/wiki/Algoritmo_de_Bresenham
        
        return puntos;
    }
    
}
