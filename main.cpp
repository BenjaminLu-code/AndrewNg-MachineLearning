#include <iostream>
#include <fstream>
#include <string>
class Person {
public:
	void operator()(std::string name) {
		std::cout << name << std::endl;
	}
};
void test01() {
	Person p;
	p("Hello world!");
	Person()("Hello world!");
}
int main() {
	test01();
}





