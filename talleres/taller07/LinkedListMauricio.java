import java.lang.IndexOutOfBoundsException; // Usar esto cuando se salga el índice
// Una lista simplemente enlazada

public class LinkedListMauricio {

  private Node first;
  private int size;

  public LinkedListMauricio() {
    size = 0;
    first = null;
  }

  /**
   * Returns the node at the specified position in this list.
   * 
   * @param index - index of the node to return
   * @return the node at the specified position in this list
   * @throws IndexOutOfBoundsException
   */
  private Node getNode(int index) throws IndexOutOfBoundsException {
    if (index >= 0 && index < size) {
      Node temp = first;
      for (int i = 0; i < index; i++) {
        temp = temp.next;
      }
      return temp;
    } else {
      throw new IndexOutOfBoundsException();
    }
  }

  /**
   * Returns the element at the specified position in this list.
   * 
   * @param index - index of the element to return
   * @return the element at the specified position in this list
   * @throws IndexOutOfBoundsException
   */
  public int get(int index) throws IndexOutOfBoundsException {
    Node temp = null;
    try {
      temp = getNode(index);
    } catch (IndexOutOfBoundsException e) {
      e.printStackTrace();
      System.exit(0);
    }
    return temp.data;
  }

  // Retorna el tamaño actual de la lista
  public int size() {
    return this.size;
  }

  // Inserta un dato en la posición index
  public void insert(int data, int index) throws IndexOutOfBoundsException {
    if (size == 0) {
      Node nodo = new Node(data);
      first = nodo;
    } else if (index == 0) {
      Node nuevo = new Node(data);
      nuevo.next = getNode(0);
      first = nuevo;
      size++;
    } else if (index >= size + 1) {
      throw new IndexOutOfBoundsException();
    } else {
      Node temp = getNode(index - 1);
      Node nuevo = new Node(data);
      nuevo.next = temp.next;
      temp.next = nuevo;
      size++;
    }
  }

  // Borra el dato en la posición index
  public void remove(int index) {
    try {
      if (index == 0) {
        Node temp = first;
        first = temp.next;
        size--;
      } else if (index == size - 1) {
        Node temp = getNode(size - 2);
        temp.next = null;
        size--;
      } else {
        Node temp = getNode(index - 1);
        temp.next = temp.next.next;
        size--;
      }
    } catch (IndexOutOfBoundsException e) {
      System.out.println("La posicion no existe");
    }

  }

  // Verifica si está un dato en la lista
  public boolean contains(int data) {
    if (size == 0)
      return false;
    else {
      Node currentNode = first;
      while (currentNode.next != null) {
        if (data == currentNode.data) {
          return true;
        }
        currentNode = currentNode.next;
      }
      return false;
    }
  }
}
