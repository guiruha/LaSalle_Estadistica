############################################################
#
# Sesion 1 - MUBD
#
# Manejo básico de R
#
############################################################

############################################################
#
# Como pedir ayuda
#
############################################################

?mean                         # interrogante para pedir ayuda de una funcion --> equivalente a help("mean")
??geometric                   # doble interrogante para pedir ayuda general --> equivalente a help.search("geometric")

############################################################
#
# R como calculadora
#
############################################################
2+3               # suma
3*3               # multiplicación
2/5               # división
3^2               # potencia
19%/%6            # cociente entero
19%%6             # resto

# Caso de uso para el cociente entero
# Tenemos fechas que vienen en formate año semana (e.g. 202620)
# Cómo separamos el año de la semana de forma rápida?
202620 %/% 100
202620 %% 100


pi                # valor de pi
letters           # letras minusculas
LETTERS           # letras mayusculas
month.name        # nombre de los meses


sqrt(81)
exp(0)
log(1)
log10(10)
abs(-10)
factorial(3)
choose(5, 3)

############################################################
#
# Variables
#
############################################################

nota <- sum(2,3)

nota / 2

############################################################
#
# Funciones
#
############################################################

plot(5, 10, "h")

cuadrado <- function(x){x^2}

cuadrado(3)

##-- Ejercicio 1.1:

# Productos vendidos por dia
# Una tienda ha vendido los siguientes productos durante tres días:
  
# Lunes: 15 productos.
# Martes: 22 productos.
# Miércoles: 18 productos.

# Cada producto tiene un precio de venta de 12 € y un coste de 7 €.

# Realiza las siguientes operaciones en R:
  
# 1. Crear las variables lunes, martes y miercoles, asignándoles el número de productos vendidos en cada día.
# 2. Crear las variables precio y coste, asignándoles los valores correspondientes.
# 3. Calcular el número total de productos vendidos durante los tres días.
# 4. Calcular la media diaria de productos vendidos.
# 5. Calcular los ingresos totales obtenidos durante los tres días.
# 6. Calcular el beneficio total, considerando la diferencia entre ingresos y costes.
# 7. Crear una función llamada beneficio que permita calcular el beneficio obtenido 
# a partir del número de unidades vendidas en un día.
# 8. Aplicar la función a las ventas del lunes y del martes. 

############################################################
#
# Tipos de objetos
#
############################################################

############################################################
# Vectores: serie ordenada de valores del mismo tipo --> c()
############################################################

##-- Ejemplos
v1 <- c(32,40,61,71,39)       # edades de personas (vector numerico)
v1
v2 <- 1:5                     # enteros del 1 al 5 (vector entero)
v2
v3 <- c("h","m","h","m","m")  # genero de personas (vector de caracteres)
v3
class(v1)
class(v2)
class(v3)

##-- Acceso y modificacion
## NOTA: Los índices empieza en 1 en R.
v1[3]          # Ver 3r elemento
v1[3] <- 62    # Modificar 3r elemento
v1[3]          # Ver 3r elemento

v1[2:4]        # Ver los elementos del 2 al 4
v1[c(1,2,4)]   # Ver los elementos 1,2 y 4

############################################################
# Matrices: serie de columnas numericas ordenadas y de igual longitud --> matrix 
############################################################
##-- Ejemplos
m1 <- matrix(c(10,20,30,40),nrow=2,byrow=TRUE)
m1

m2 <- cbind(v1,v2) # Para filas: rbind
m2

##-- Acceso y modificacion
m1[2,1]       # Elemento de la fila 2, columna 1       
m1[2,]        # Elementos de la fila 2
m1[,1]        # Elementos de la columna 1
m1[2,1] <- 35 # Modificar el elemento de la fila 2, columna 1
m1

colnames(m2) <- c("edad","orden")  # Modificar el nombre de las columnas 
rownames(m2) <- letters[1:5]       # Modificar el nombre de las filas
colnames(m2)                       # Consultar el nombre de las columnas 
rownames(m2)                       # Consultar el nombre de las filas
m2

############################################################
# Data.frames: serie de columnas de cualquier tipo ordenadas y de igual longitud --> data.frame
############################################################
##-- Ejemplos
df1 <- data.frame(x=letters[1:10],y=1:10)   # Letras del 1 al 10 y numeros del 1 al 10
View(df1)
df2 <- data.frame(v3,v1)
df2

##-- Acceso y modificacion
df1$x          # Columna llamada "x"
df1[,"x"]      # Columna llamada "x" (alternativa a la anterior opcion)
df1[,1]        # Primera columna     (alternativa a la anterior opcion)

df1[2,1]       # Elemento de la fila 2, columna 1
df1$x[2]       # Elemento de la fila 2, columna 1 (alternativa a la anterior opcion)

df1[2,]        # Elementos de la fila 2
df1[2,2] <- 35 # Modificar el elemento de la fila 2, columna 2
df1

df1[3,1] <- "H" # Modificar el elemento de la fila 3, columna 1.
df1

## Funciones de búsqueda útiles
which(df1$x=='e')
which.max(df1$y)
which.min(df1$y)
subset(df1, x=='e')
subset(df1, y==max(df1$y))


############################################################
# Listas: cualquier combinacion de los elementos anteriores --> list 
############################################################
##-- Ejemplos 
l1 <- list(v1,m1,df1)
l1

##-- Acceso y modificacion
l1[1]               # Un corchete para simplificar la lista
class(l1[1])

l1[[1]]             # Doble corchete para acceder al elemento
class(l1[[1]])

l1[[4]] <- c(1:10)

l1[[4]] <- 4


##-- Ejercicio 1.2:
# 1. Crear un data.frame con los siguientes datos
#     marca modelo consumo
#      audi     A3     6.8
# volswagen   Polo     6.8
#       bmw     X5    12.5
#   renault megane     4.7
#      fiat  punto     5.0
# 2. Modificar el consumo del fiat punto a 4.5


############################################################
#
# EXTRA: Instrucciones utiles
#
############################################################

##-- Programacion

if(v1[1]==32) print("La edad del primer individuo es 32")  # if --> condicion logica
for(i in 1:5) print(v1[i])                                 # Bucle 


##-- Gestion de objetos
ls()              # Objetos en memoria
object.size(v1)   # Tamaño que ocupa un objeto en memoria
rm(v1)            # Borrar (remove) algun objeto en memoria
rm(list=ls())     # Borrar todos los objetos en memoria


############################################################
#
# Lectura de datos
#
############################################################
##-- De un fichero de texto
setwd('~/Desktop/LaSalle_Estadistica/Sesion1/Practica/Datos/')
datos <- read.table('Mobiles.txt',header=TRUE,sep=";",dec=".",na.strings="NA", stringsAsFactors = TRUE)

##-- Inspeccion de los datos
dim(datos)               # numero de filas y de columnas
head(datos)              # cabecera de los datos
summary(datos)           # descriptiva de los datos

##-- Hay missings no codificados correctamente. Se repite la lectura
datos <- read.table('Mobiles.txt',header=TRUE,sep=";",dec=".",na.strings=c("NA",""), stringsAsFactors = TRUE)
summary(datos)

##-- Ejercicio 1.3:
# 1. Leer el fichero flights.txt cambiando los parametros de lectura oportunos y guardar en la variable datos2 
# NOTA: (OJO con el simbolo de los missings)
# 2. Inspeccionar los datos: num. filas y columnas, visualizar los datos y descriptiva global
# 3. ¿Cuanto tiempo estuvo en el aire el vuelo 251 y con numero de cola N855UA?

############################################################
#
# Escritura de datos a un fichero
#
############################################################
##-- A un fichero de texto
write.table(datos,"Mobiles_test.txt",quote=FALSE,row.names = FALSE,sep="\t")

##-- Copiar el subconjunto de moviles Nokia a un fichero llamado Nokia_test
write.table(subset(datos,Brand=="Nokia"),"Nokia_test.txt",quote=FALSE,row.names = FALSE,sep="\t")


##-- Ejercicio 1.4:
# 1. Copiar los vuelos que estuvieron volando menos de 1 hora en un fichero llamado short_flights_test.txt
# [El fichero sin usar comillas para las categorias,sin nombre de columnas ni de filas y columnas separadas por tabuladores]


############################################################
#
# Descriptiva Univariante
#
############################################################

# install.packages("MindOnStats")
# library(MindOnStats)

# data(Mobiles)
# datos <- Mobiles

datos <- read.table('Mobiles.txt',header=TRUE,sep=";",dec=".",na.strings=c("NA", ""), stringsAsFactors = TRUE)

dim(datos)
colnames(datos)
summary(datos)

############################################################
# Variable numerica
############################################################
##-- Estadisticos principales de una variable
mean(datos$Age)

var(datos$Age)

sd(datos$Bill)                       # atencion: no funciona por tener missings
sd(datos$Bill,na.rm=TRUE)            # se deben eliminar los missings

summary(datos$Bill)                  

##-- ¿Que representan los cuartiles (1st Qu.,Median(2nd Qu.),3rd Qu.)?
n <- dim(datos)[1]                   # numero de observaciones

sum(datos$Bill<=25.00,na.rm=TRUE)/n  # proporcion de observaciones por debajo del 1st Qu.
sum(datos$Bill<=35.00,na.rm=TRUE)/n  # proporcion de observaciones por debajo del 2nd Qu.
sum(datos$Bill<=60.00,na.rm=TRUE)/n  # proporcion de observaciones por debajo del 3rd Qu.

##-- IQR
(iqr <- IQR(datos$Bill,na.rm=TRUE))  # rango intercuartilico: distancia entre Q1 y Q3

##-- Graficos
 
boxplot(datos$Bill, col="blue", main="Factura", horizontal=TRUE)
##-- ?Que representan las lineas del boxplot?
abline(v=25, lty=2, col=1)             # Primer cuartil
abline(v=35, lty=2, col=2)             # Mediana
abline(v=60, lty=2, col=3)             # Tercer cuartil
abline(v=0, lty=2, col=4)              # Bigote inferior (si es 0)
abline(v=60+1.5*iqr, lty=2, col=5)     # Límite superior de valores atípicos


hist(datos$Age,col="navy",freq=FALSE)

# Densidad
(sum(datos$Age > 20 & datos$Age < 25)/dim(datos)[1])/5

############################################################
# Variable categorica
############################################################
##-- Tabla de frecuencias y proporciones
table(datos$Brand)                           # Tabla de frecuencias
prop.table(table(datos$Brand))               # Tabla de proporciones

round(100*prop.table(table(datos$Brand)),1)  # Tabla de porcentajes redondeados

# Ver https://dabblingwithdata.wordpress.com/2017/12/20/my-favourite-r-package-for-frequency-tables/ para alternativas

##-- Diagrama de barras
barplot(table(datos$Brand))                  

# Diagrama de barras mejorado
quartz()# windows()
barplot(sort(table(datos$Brand)),las=2)

##-- Ejercicio 1.5
# 1. Realiza la descriptiva de la variable tiempo de vuelo (air_time) del conjunto de datos flights.txt

# 2. Realiza la descriptiva de la variable categorica origen (origin) del conjunto de datos flights.txt

############################################################
#
# Descriptiva bivariante
#
############################################################

############################################################
# Numerica vs Numerica
############################################################
##-- Correlacion: relacion lineal entre variables (entre -1 y 1)
cor(datos$Age,datos$Bill,use="complete.obs")    # alternativa 1
with(datos,cor(Age,Bill,use="complete.obs"))    # alternativa 2

##-- Grafico bivariante (hacer ?par para ver parametros)
plot(datos$Age,datos$Bill)

plot(datos$Age,datos$Bill,xlab="Edad",ylab="Factura",main="Relacion de Edad y Gasto")

plot(Bill~Age,datos,xlab="Edad",ylab="Factura",main="Relacion de Edad y Gasto",
     col=4,pch=19,cex=0.8,las=1)                                                    

plot(jitter(log(Bill+0.1))~jitter(log(Age)),datos,  # OJO al log(X+1)
     col=4,pch=19,cex=0.8,las=1,
     xlab="Edad",ylab="Factura",main="Relacion de Edad y Gasto")  

############################################################
# Numerica vs Categorica
############################################################

############################################################
# Categorica vs Categorica
############################################################
##-- Descriptiva categorica vs categorica: table y prop.table
with(datos,table(PlanType,PrimaryUse))               # Frecuencias

with(datos,prop.table(table(PlanType,PrimaryUse)))   # Proporciones globales

with(datos,prop.table(table(PlanType,PrimaryUse),1)) # Proporciones por filas (condicionada)

with(datos,prop.table(table(PlanType,PrimaryUse),2)) # Proporciones por columnas (condicionada)

##-- Barplot y mosaicplot
quartz(10, 7) # Para Windows: windows(10,7)

par(mfrow=c(1,2),las=1)

with(datos,barplot(table(PrimaryUse,PlanType),col=1:3,legend=TRUE,args.legend=list(x='topleft')))
with(datos,
 barplot(table(PrimaryUse, PlanType),
   col = 1:3,
   legend = TRUE,
   beside = TRUE,
   args.legend = list(x = "topleft"))
)


par(mfrow=c(1,1),las=1)
mosaicplot(PlanType~PrimaryUse,datos,col=1:3,main="Uso segun uso de contrato")

##-- Descriptiva numerica estratificada: tapply
summary(datos$Bill)                             # Descriptiva global
tapply(datos$Bill,datos$PlanType,summary)       # Descriptiva estratificada segun plan
tapply(datos$Bill,datos$PlanType,sd,na.rm=TRUE) # Desviacion segun plan

##-- Boxplot estratificado
boxplot(Bill~PlanType,datos,col="steelblue",main="Factura según plan")

##-- Ejercicio 1.6
# 1. Realiza la descriptiva del retraso en la llegada segun el retraso en la salida. 
# Calcula la correlacion y haz el diagrama bivariante. Tambien usa la instruccion smoothScatter para hacer un plot alternativo 
# 
# 2. Realiza la descriptiva del retraso en la llegada segun el mes del anyo
# Haz el summary del retraso segun el mes y el boxplot estratificado
#
# Crea un subconjunto de datos2 llamado datos2.delay para aquellos vuelos con retraso en la llegada
# Vuelve a hacer la descriptiva anterior
# 
# 
# 3. Realiza la descriptiva de la hora de llegada segun la hora de salida (hour) como si fuese una categorica  
# 
# Crea una nueva variable dentro de datos 2 que sea una categorizacion de la variable hour en 3 categorias: [0-8],[9-16],[17-24]
# Repite la descriptiva anterior (mira la ayuda de la instruccion ?cut)
