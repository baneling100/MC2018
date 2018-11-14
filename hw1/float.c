/**
 * Multicore Computing Assignment #1 Problem #1
 * Date: 2018-09-23
 * Name: Baek, Woohyeon
 * Student ID: 2017-15782
 * Department: Computer Science and Engineering
 */

#include <stdio.h>

int main() {
	int i;
	float num;
	int *ptr;

	printf("Enter a number: ");
	scanf("%f", &num);
	ptr = &num;

	for(i = 31; i >= 0; i--)
		printf("%d", ((*ptr) >> i) & 1);
	printf("\n%f\n", num);

	return 0;
}

