#include <iostream>
// a simple example of constructor and deconstructor
// we create a automatic memory cleaner, whenever we create a Buffer obejct, (ex. Buffer buf(5)) 
// it creates a array of size 'size' dynamicaally on a heap, and then automarically when the work is done, 
// the fucntion completes the execution, it simply delete the local object. 
class Buffer{
    int* data;
    public:
    Buffer(int size) {data = new int[size]; } // constructor
    ~Buffer() { delete[] data; } // runs after when the Buffer goes out of scope , i.e. deconstructor
};