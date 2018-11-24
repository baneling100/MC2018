TARGET = mat_mul_ref

MODE = single
DEVICE = gpu/7970

NDIM = 16384
MDIM = 16384
EXE_OPTION = 

.PHONY: clean run

clean:
	rm -rf $(TARGET) $(OBJECTS) task*

run: $(TARGET)
	@thorq --add --mode $(MODE) --device $(DEVICE) --timeout 60 ./$(TARGET) $(NDIM) $(MDIM) $(EXE_OPTION)
