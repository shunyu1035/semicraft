// film.h
typedef struct {
    int id;
    int index[3];
    double normal[3];
} Cell;

typedef struct {
    int f1; // Si
    int f2; // SiCl
    int f3; // SiCl2
    int f4; // SiCl3
    int f5; // Mask
} Film;
