package language.generics;

public class MyArray<E> {
    Object[] elements;
    int index;

    public MyArray(int size) {
        index = 0;
        elements = new Object[size];
    }

    public void add(E element) {
        elements[index++] = element;
    }

    public E get(int pos) {
        return (E) elements[pos];
    }
}
