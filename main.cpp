#include "map.h"
#include "localmapping.h"
#include "visualize.h"

int main()
{
	Map* map = new Map();
	LocalMapping* mapping = new LocalMapping(map);
	mapping->Run();
	Visualize::VisualizeMap(mapping->GetMap());
    return 0;    
}