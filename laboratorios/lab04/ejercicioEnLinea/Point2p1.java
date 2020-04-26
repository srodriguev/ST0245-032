// Preorder > Inorder
// Código ensamblado bajo referencia de los métodos de GeeksForGeeks, por Mayank Jaiswal 
 //Origen: https://www.geeksforgeeks.org/construct-bst-from-given-preorder-traversa/?ref=lbp  

  
import java.util.*; 

// Noditos del binary tree
class Node { 
  
    int data; 
    Node left, right; 
  
    Node(int d) { 
        data = d; 
        left = right = null; 
    } 
} 
  
public class Point2p1 { 
  
    // Función que construye desde el pre[] 
    Node construirArbol(int pre[], int size) { 
  
        // El primer elemento es la raíz 
        Node root = new Node(pre[0]);  
        // Stack para guardar los datos 
        Stack<Node> s = new Stack<Node>(); 
  
        // Push la raíz 
        s.push(root); //push O(n)
  
        // itera en los demás items del array
        for (int i = 1; i < size; ++i) { 
            Node temp = null; 
  
            /* Sigue haciendo pops mientras que el próximo valor sea mayor que el top del stack */
            while (!s.isEmpty() && pre[i] > s.peek().data) 
            { 
                temp = s.pop(); 
            } 
  
            // Hace este valor mas grande el hijo de la derecha y lo empuja al stack
            if (temp != null) 
            { 
                temp.right = new Node(pre[i]); 
                s.push(temp.right); //push O(n)
            }  
              
            /* Si el próximo valor es menor que el top del stack este valor se convierte en el hijo izquierdo del nodo tope del stack.
            Lo empuja al stack. */
            else 
            { 
                temp = s.peek(); 
                temp.left = new Node(pre[i]); 
                s.push(temp.left); //push O(n)
            } 
        } 
  
        return root; 
    } 
  
    // Funcón secundaria de impresión
    void imprimirInorder(Node node) 
    { 
        if (node == null) 
        { 
            return; 
        } 
        imprimirInorder(node.left); 
        System.out.print(node.data + " "); 
        imprimirInorder(node.right); 
    } 
  
    // Método main de pruebas.
    public static void main(String[] args) 
    { 
        Point2p1 tree = new Point2p1(); 
        int pre[] = new int[]{50,30,24,5,28,45,98,52,60}; 
        int size = pre.length; 
        Node root = tree.construirArbol(pre, size); 
        System.out.println("Inorder traversal: "); 
        tree.imprimirInorder(root); 
    } 
} 