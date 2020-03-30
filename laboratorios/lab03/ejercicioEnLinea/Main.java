import java.util.LinkedList;
class Main {
  public static void main(String[] args) {

    String a = arreglo("This_is_a_[Beiju]_text");
    String b = arreglo("[[]][][]Happy_Birthday_to_Tsinghua_University");
    String c = arreglo("asd[fgh[jkl");
    String d = arreglo("asd[fgh[jkl[");
    String e = arreglo("[[a[[d[f[[g[g[h[h[dgd[fgsfa[f");

    System.out.println(a.equals("BeijuThis_is_a__text"));
    System.out.println(b.equals("Happy_Birthday_to_Tsinghua_University"));
    System.out.println(c.equals("jklfghasd"));
    System.out.println(d.equals("jklfghasd"));
    System.out.println(e.equals("ffgsfadgdhhggfda"));
  }

  public static String arreglo(String entrada) {
    LinkedList<Character> texto = new LinkedList<>();
    int cont=0;
    boolean inicio=false;
    String res="";
    
    for(int i=0; i<entrada.length();i++){ // O(n)
      char c=entrada.charAt(i);

      if(c=='[' && i<entrada.length()-1){
        inicio=true;
        i++;
        cont=0;
      }
      else if(c==']' && i<entrada.length()-1){
        inicio=false;
        i++;
      }

      if(inicio){
        texto.add(cont,entrada.charAt(i));
        cont++;
      }
      else{
        texto.add(entrada.charAt(i));
      }
    }

    LinkedList<Character> cor = new LinkedList<>();
    cor.add('[');
    cor.add(']');
    texto.removeAll(cor); //O(n)

    for(int i=0;i<texto.size();i++){ // O(n)
      res+=texto.get(i);
    }
    return res ;
  }
}