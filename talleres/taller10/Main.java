class Main {
  public static void main(String[] args) {
    System.out.println("Ejercicio de árboles binarios: ");

    BinarySearchTree tree = new BinarySearchTree(8);
    tree.insertar(4);
    tree.insertar(2);
    tree.insertar(7);
    tree.insertar(10);
    tree.insertar(12);


    boolean bool12 = tree.buscar(12);
    System.out.println(bool12);
    tree.borrar(12);
    bool12 = tree.buscar(12);
    System.out.println(bool12);
    tree.imprimirarbol(tree);
  }
}