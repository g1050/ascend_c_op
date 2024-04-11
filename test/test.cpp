#include <iostream>
#include <vector>
// 普通的模板类定义
template<typename T>
class Converter {
public:
    // 默认情况下的转换函数
    T convert(const T& input) {
        std::cout << "Using default conversion: ";
        return input;
    }
    void hello(){
        std::cout << "hello world parent";
    }
    int a = 10;
    std::vector<T> v;
};

template<typename T>
class myClass : public Converter<T>{
    public:
    void hello(){
        std::cout << "hello world parent";
        this->a = 12;
        Converter<T>::a = 12;
        this->v.size();
    }
};
// 对特定类型的模板类进行特化
template<>
class Converter<int> {
public:
    int convert(const int& input) {
        std::cout << "Specialized conversion for int: ";
        return input * 2;
    }
};

template<>
class Converter<double> {
private:
    std::string name;
public:
    double convert(const double& input) {
        name = "gxk";
        std::cout << "Specialized conversion for double: " << name ;
        return input * 1.5;
    }
};

class NewConverter: public Converter<uint>{
    public:
    void hello(){
        std::cout << "hello world NewConverter";
    }
    uint convert(const uint& input) {
        Converter<uint>::hello();
        Converter<uint>::convert(input);
        std::cout << "Specialized conversion for uint: ";
        return input * 2;
    }
};

int main() {
    // 使用模板类
    Converter<int> intConverter;
    int intResult = intConverter.convert(5);
    std::cout << intResult << std::endl;

    Converter<double> doubleConverter;
    double doubleResult = doubleConverter.convert(3.14);
    std::cout << doubleResult << std::endl;

    NewConverter newConverter;
    uint uintResult = newConverter.convert(3.14);
    std::cout << uintResult << std::endl;

    myClass<int> my;

    // 对于其他类型，使用默认转换
    Converter<std::string> stringConverter;
    std::string stringResult = stringConverter.convert("Hello");
    std::cout << "Using default conversion: " << stringResult << std::endl;

    return 0;
}
