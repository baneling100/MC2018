/**
 * Multicore Computing Assignment #1 Problem #2
 * Date: 2018-09-24
 * Name: Baek, Woohyeon
 * Student ID: 2017-15782
 * Department: Computer Science and Engineering
 */

#include <stdio.h>
#include <sys/time.h>
#define MILLION 1000000.0

int main() {

	int i;
	struct timeval begin, end;
	double time_spent;

	for(i = 0; i < 100000000; i++) {
		__asm__ __volatile__("vaddss %xmm0,%xmm0,%xmm0");
		__asm__ __volatile__("vaddss %xmm1,%xmm1,%xmm1");
		__asm__ __volatile__("vaddss %xmm2,%xmm2,%xmm2");
		__asm__ __volatile__("vaddss %xmm3,%xmm3,%xmm3");
		__asm__ __volatile__("vaddss %xmm4,%xmm4,%xmm4");
		__asm__ __volatile__("vaddss %xmm5,%xmm5,%xmm5");
		__asm__ __volatile__("vaddss %xmm6,%xmm6,%xmm6");
		__asm__ __volatile__("vaddss %xmm7,%xmm7,%xmm7");
		__asm__ __volatile__("vaddss %xmm8,%xmm8,%xmm8");
		__asm__ __volatile__("vaddss %xmm9,%xmm9,%xmm9");
		__asm__ __volatile__("vaddss %xmm10,%xmm10,%xmm10");
		__asm__ __volatile__("vaddss %xmm11,%xmm11,%xmm11");
		__asm__ __volatile__("vaddss %xmm12,%xmm12,%xmm12");
		__asm__ __volatile__("vaddss %xmm13,%xmm13,%xmm13");
		__asm__ __volatile__("vaddss %xmm14,%xmm14,%xmm14");
		__asm__ __volatile__("vaddss %xmm15,%xmm15,%xmm15");
	}
	
	gettimeofday(&begin, NULL);
	for(i = 0; i < 100000000; i++) {
		__asm__ __volatile__("vaddss %xmm0,%xmm0,%xmm0");
		__asm__ __volatile__("vaddss %xmm1,%xmm1,%xmm1");
		__asm__ __volatile__("vaddss %xmm2,%xmm2,%xmm2");
		__asm__ __volatile__("vaddss %xmm3,%xmm3,%xmm3");
		__asm__ __volatile__("vaddss %xmm4,%xmm4,%xmm4");
		__asm__ __volatile__("vaddss %xmm5,%xmm5,%xmm5");
		__asm__ __volatile__("vaddss %xmm6,%xmm6,%xmm6");
		__asm__ __volatile__("vaddss %xmm7,%xmm7,%xmm7");
		__asm__ __volatile__("vaddss %xmm8,%xmm8,%xmm8");
		__asm__ __volatile__("vaddss %xmm9,%xmm9,%xmm9");
		__asm__ __volatile__("vaddss %xmm10,%xmm10,%xmm10");
		__asm__ __volatile__("vaddss %xmm11,%xmm11,%xmm11");
		__asm__ __volatile__("vaddss %xmm12,%xmm12,%xmm12");
		__asm__ __volatile__("vaddss %xmm13,%xmm13,%xmm13");
		__asm__ __volatile__("vaddss %xmm14,%xmm14,%xmm14");
		__asm__ __volatile__("vaddss %xmm15,%xmm15,%xmm15");
	}
	gettimeofday(&end, NULL);
	time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / MILLION;
	printf("float addition time : %lf\n", time_spent);
	printf("float addition flops: %lf\n", 1600000000 / time_spent);
	
	gettimeofday(&begin, NULL);
	for(i = 0; i < 100000000; i++) {
		__asm__ __volatile__("vmulss %xmm0,%xmm0,%xmm0");
		__asm__ __volatile__("vmulss %xmm1,%xmm1,%xmm1");
		__asm__ __volatile__("vmulss %xmm2,%xmm2,%xmm2");
		__asm__ __volatile__("vmulss %xmm3,%xmm3,%xmm3");
		__asm__ __volatile__("vmulss %xmm4,%xmm4,%xmm4");
		__asm__ __volatile__("vmulss %xmm5,%xmm5,%xmm5");
		__asm__ __volatile__("vmulss %xmm6,%xmm6,%xmm6");
		__asm__ __volatile__("vmulss %xmm7,%xmm7,%xmm7");
		__asm__ __volatile__("vmulss %xmm8,%xmm8,%xmm8");
		__asm__ __volatile__("vmulss %xmm9,%xmm9,%xmm9");
		__asm__ __volatile__("vmulss %xmm10,%xmm10,%xmm10");
		__asm__ __volatile__("vmulss %xmm11,%xmm11,%xmm11");
		__asm__ __volatile__("vmulss %xmm12,%xmm12,%xmm12");
		__asm__ __volatile__("vmulss %xmm13,%xmm13,%xmm13");
		__asm__ __volatile__("vmulss %xmm14,%xmm14,%xmm14");
		__asm__ __volatile__("vmulss %xmm15,%xmm15,%xmm15");
	}
	gettimeofday(&end, NULL);
	time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / MILLION;
	printf("float multiplication time : %lf\n", time_spent);
	printf("float multiplication flops: %lf\n", 1600000000 / time_spent);
	
	gettimeofday(&begin, NULL);
	for(i = 0; i < 100000000; i++) {
		__asm__ __volatile__("vaddsd %xmm0,%xmm0,%xmm0");
		__asm__ __volatile__("vaddsd %xmm1,%xmm1,%xmm1");
		__asm__ __volatile__("vaddsd %xmm2,%xmm2,%xmm2");
		__asm__ __volatile__("vaddsd %xmm3,%xmm3,%xmm3");
		__asm__ __volatile__("vaddsd %xmm4,%xmm4,%xmm4");
		__asm__ __volatile__("vaddsd %xmm5,%xmm5,%xmm5");
		__asm__ __volatile__("vaddsd %xmm6,%xmm6,%xmm6");
		__asm__ __volatile__("vaddsd %xmm7,%xmm7,%xmm7");
		__asm__ __volatile__("vaddsd %xmm8,%xmm8,%xmm8");
		__asm__ __volatile__("vaddsd %xmm9,%xmm9,%xmm9");
		__asm__ __volatile__("vaddsd %xmm10,%xmm10,%xmm10");
		__asm__ __volatile__("vaddsd %xmm11,%xmm11,%xmm11");
		__asm__ __volatile__("vaddsd %xmm12,%xmm12,%xmm12");
		__asm__ __volatile__("vaddsd %xmm13,%xmm13,%xmm13");
		__asm__ __volatile__("vaddsd %xmm14,%xmm14,%xmm14");
		__asm__ __volatile__("vaddsd %xmm15,%xmm15,%xmm15");
	}
	gettimeofday(&end, NULL);
	time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / MILLION;
	printf("double addition time : %lf\n", time_spent);
	printf("double addition flops: %lf\n", 1600000000 / time_spent);
	
	gettimeofday(&begin, NULL);
	for(i = 0; i < 100000000; i++) {
		__asm__ __volatile__("vmulsd %xmm0,%xmm0,%xmm0");
		__asm__ __volatile__("vmulsd %xmm1,%xmm1,%xmm1");
		__asm__ __volatile__("vmulsd %xmm2,%xmm2,%xmm2");
		__asm__ __volatile__("vmulsd %xmm3,%xmm3,%xmm3");
		__asm__ __volatile__("vmulsd %xmm4,%xmm4,%xmm4");
		__asm__ __volatile__("vmulsd %xmm5,%xmm5,%xmm5");
		__asm__ __volatile__("vmulsd %xmm6,%xmm6,%xmm6");
		__asm__ __volatile__("vmulsd %xmm7,%xmm7,%xmm7");
		__asm__ __volatile__("vmulsd %xmm8,%xmm8,%xmm8");
		__asm__ __volatile__("vmulsd %xmm9,%xmm9,%xmm9");
		__asm__ __volatile__("vmulsd %xmm10,%xmm10,%xmm10");
		__asm__ __volatile__("vmulsd %xmm11,%xmm11,%xmm11");
		__asm__ __volatile__("vmulsd %xmm12,%xmm12,%xmm12");
		__asm__ __volatile__("vmulsd %xmm13,%xmm13,%xmm13");
		__asm__ __volatile__("vmulsd %xmm14,%xmm14,%xmm14");
		__asm__ __volatile__("vmulsd %xmm15,%xmm15,%xmm15");
	}
	gettimeofday(&end, NULL);
	time_spent = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / MILLION;
	printf("double multiplication time : %lf\n", time_spent);
	printf("double multiplication flops: %lf\n", 1600000000 / time_spent);
	
	return 0;
}

