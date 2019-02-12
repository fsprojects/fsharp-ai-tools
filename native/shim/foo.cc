#include "foo.h"
//
//class Foo {
//public:
//  int Bar();
//};
//
//int Foo::Bar() {
//	std::cout << "foobar" << std::endl;
//	return 1;
//};


extern "C" {
	int TwoTimes(int x) { return 2 * x; }
	void TwoTimes2(int x, Callback call) {
		call(x, 2 * x);
	}

//	Foo* Foo_Create() { return new Foo(); }
//	int Foo_Bar(Foo* pFoo) { return pFoo->Bar(); }
//	void Foo_Delete(Foo* pFoo) { delete pFoo; }
}

//int main() {
//	std::cout << "fooo" << std::endl;
//	std::string name;
//	std::getline(std::cin, name);
//}

