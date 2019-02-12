
#define DLLEXTERN __declspec(dllexport)

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

typedef int(__stdcall* Callback)(
	int x,
	int y);

extern "C" {
	DLLEXTERN int TwoTimes(int x);
	DLLEXTERN void TwoTimes2(int x, Callback call);

//	Foo* Foo_Create() { return new Foo(); }
//	int Foo_Bar(Foo* pFoo) { return pFoo->Bar(); }
//	void Foo_Delete(Foo* pFoo) { delete pFoo; }
}

//int main() {
//	std::cout << "fooo" << std::endl;
//	std::string name;
//	std::getline(std::cin, name);
//}