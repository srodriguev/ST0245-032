public class Main {

    public static void main(String[] args) 
    {
        System.out.println("Ejericio contador: ");
        Contador c = new Contador("test"); 
        for (int i = 0; i < 10; ++i)
            c.incrementar(1);
            // 10
        System.out.println(c);

        System.out.println();
        Punto p = new Punto(-1, 1);
        // 0,0
        System.out.printf("Punto: {%f, %f}\n", p.x(), p.y());
        // 1*sqrt(2)
        System.out.println("Radio Polar: " + p.radioPolar());
        // -45
        System.out.println("Angulo Polar en grados: " + Math.toDegrees(p.anguloPolar()));
        // 2
        System.out.println("Dist Euclidiana: " + p.distanciaEuclidiana(new Punto(1, 1)));
        
        System.out.println();
        Punto p2 = new Punto(10,20);
        System.out.printf("Punto: {%f, %f}\n", p2.x(), p2.y());
        System.out.println("Radio Polar: " + p2.radioPolar());
        System.out.println("Angulo Polar en grados " + Math.toDegrees(p2.anguloPolar()));
        System.out.println("Dist Euclidiana: " + p2.distanciaEuclidiana(new Punto(0, 0)));

        System.out.println();

        Fecha f1 = new Fecha(1, 8, 2017);
        Fecha f2 = new Fecha(2, 5, 2016);
        Fecha f3 = new Fecha(1, 8, 2017);
        System.out.println(f1);
        System.out.println(f2);
        System.out.println(f3);
        // 1
        System.out.println(f1.comparar(f2));
        System.out.println(f1.comparar(f3));
    }

}
