public class Taller3 {

	
	
	public static void torresDeHannoi(int n) {
		torresDeHannoiAux(n, 1, 2, 3);
	}


	
	private static void torresDeHannoiAux(int n, int origen, int intermedio, int destino) {
        if(n==1){
            System.out.println("disco "+n+" de "+origen+" a "+destino);
        }
        else{
            torresDeHannoiAux(n-1,origen, destino, intermedio);
            System.out.println("disco "+n+" de "+origen+" a "+destino);
            torresDeHannoiAux(n-1, intermedio, origen, destino);
        }
	}


  public static void permutation(String str) {
		permutationAux("", str); 
	}
	
	
	private static void permutationAux(String prefix, String str) {
        if(str.length()==0 ) {
          System.out.println(prefix); 
          String s=AdvancedEncryptionStandard.desencriptarArchivo(prefix);
        System.out.println(s); }
        else{
        for(int i =0; i<str.length(); i++){
            permutationAux(prefix + str.charAt(i) , str.substring(0,i) + str.substring(i+1,str.length()));
          }
        }
  }
}