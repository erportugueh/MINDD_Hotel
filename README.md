
# Análise dos Tipos de Dados das Colunas

Abaixo, apresento uma análise detalhada das colunas no dataset, especificando o tipo de dados, o significado de cada coluna, e os valores únicos observados nas colunas categóricas.

## Dados Numéricos

1. **`is_canceled`**: Indicador binário (0 ou 1) que identifica se a reserva foi cancelada.
2. **`lead_time`**: Número de dias desde a reserva até a data de chegada. Valores mais altos podem indicar maior probabilidade de cancelamento.
3. **`arrival_date_year`**: Ano de chegada, pode indicar tendências de cancelamento ao longo do tempo.
4. **`arrival_date_week_number`**: Semana do ano em que o hóspede chega, possível relação com períodos de alta demanda.
5. **`arrival_date_day_of_month`**: Dia do mês da chegada.
6. **`stays_in_weekend_nights`**: Quantidade de noites de fim de semana reservadas.
7. **`stays_in_week_nights`**: Quantidade de noites durante a semana reservadas.
8. **`adults`**: Número de adultos na reserva.
9. **`children`**: Número de crianças na reserva.
10. **`babies`**: Número de bebês na reserva.
11. **`previous_cancellations`**: Número de reservas anteriores canceladas pelo hóspede.
12. **`previous_bookings_not_canceled`**: Número de reservas anteriores não canceladas.
13. **`booking_changes`**: Número de mudanças feitas na reserva.
14. **`days_in_waiting_list`**: Número de dias em que a reserva permaneceu na lista de espera.
15. **`adr`**: Taxa média diária, calculada dividindo o custo total pela quantidade de noites.
16. **`required_car_parking_spaces`**: Número de vagas de estacionamento solicitadas.
17. **`total_of_special_requests`**: Quantidade de pedidos especiais feitos pelo hóspede.

## Dados Categóricos

As colunas categóricas incluem informações que descrevem características da reserva e do hóspede. Abaixo estão as descrições de cada coluna, seguidas dos valores únicos principais.

1. **`hotel`**: Tipo de hotel reservado.
   - Valores: `'Resort Hotel'`, `'City Hotel'`

2. **`arrival_date_month`**: Mês da data de chegada, com grande relevância para identificar sazonalidade.
   - Valores: `'July'`, `'August'`, `'September'`, `'October'`, `'November'`, `'December'`, `'January'`, `'February'`, `'March'`, `'April'`

3. **`meal`**: Tipo de refeição incluída na reserva.
   - Valores: `'BB'` (Bed & Breakfast), `'FB'` (Full Board), `'HB'` (Half Board), `'SC'` (Self-Catering), `'Undefined'`

4. **`country`**: Código de país do hóspede, útil para identificar a distribuição geográfica dos clientes.
   - Valores: `'PRT'` (Portugal), `'GBR'` (Reino Unido), `'USA'` (Estados Unidos), `'ESP'` (Espanha), `'IRL'` (Irlanda), `'FRA'` (França), `NaN`, `'ROU'` (Romênia), `'NOR'` (Noruega), `'OMN'` (Omã)


5. `market_segment`:
Refere-se ao canal de aquisição que originou a reserva. Os valores incluem:
    
    - **`Direct`**: Reservas feitas diretamente pelo hóspede através do site do hotel ou por telefone.
    - **`Corporate`**: Reservas feitas por empresas que têm um acordo ou contrato com o hotel.
    - **`Online TA`**: Refere-se a Agências de Viagem Online que facilitam a reserva de acomodações através de seus sites.
    - **`Offline TA/TO`**: Reservas feitas através de Agências de Viagem Tradicionais ou Operadores Turísticos.
    - **`Complementary`**: Reservas que podem ser originadas de pacotes promocionais ou colaborações com outras empresas.
    - **`Groups`**: Reservas feitas para grupos, geralmente com um número significativo de hóspedes.
    - **`Undefined`**: Reservas cujo canal de aquisição não foi especificado.
    - **`Aviation`**: Reservas originadas a partir de companhias aéreas, geralmente como parte de pacotes de viagem.


6. `distribution_channel`:
Representa o método pelo qual a reserva foi realizada.  Os valores são:

    - **`Direct`**: Reservas feitas diretamente pelo hóspede, seja pelo site ou por telefone, que geralmente oferecem melhores condições.
    - **`Corporate`**: Reservas realizadas por meio de contratos corporativos, que podem garantir uma taxa fixa ou condições especiais.
    - **`TA/TO`**: Indica reservas feitas através de Agências de Viagem ou Operadores Turísticos, que ajudam a facilitar o processo de reserva para os hóspedes.
    - **`Undefined`**: Reservas cujo canal de distribuição não foi especificado.
    - **`GDS`**: Sistema de Distribuição Global, uma plataforma que conecta hotéis e agências de viagens, facilitando a distribuição de reservas a nível global.

7. **`reserved_room_type`**: Código do tipo de quarto originalmente reservado.
   - Valores: `'C'`, `'A'`, `'D'`, `'E'`, `'G'`, `'F'`, `'H'`, `'L'`, `'P'`, `'B'`

8. **`assigned_room_type`**: Código do tipo de quarto efetivamente atribuído, que pode diferir do reservado.
   - Valores: `'C'`, `'A'`, `'D'`, `'E'`, `'G'`, `'F'`, `'I'`, `'B'`, `'H'`, `'P'`

9. **`deposit_type`**: Tipo de depósito associado à reserva, o que pode influenciar o comportamento de cancelamento.
   - Valores: `'No Deposit'`, `'Refundable'`, `'Non Refund'`

10. `customer_type`:
É uma categorização útil que permite a segmentação e análise do comportamento do hóspede. Os valores incluem:

    - **`Transient`**: Hóspedes que fazem reservas individuais, muitas vezes por lazer ou viagens de negócios rápidas.
    - **`Contract`**: Clientes que têm um contrato formal com o hotel, normalmente empresas que garantem um número fixo de quartos durante um período.
    - **`Transient-Party`**: Semelhante ao tipo 'Transient', mas refere-se a grupos que viajam juntos, como famílias ou amigos.
    - **`Group`**: Reservas feitas especificamente para grupos, como eventos corporativos, conferências ou celebrações, que podem exigir gestão e planejamento adicionais.

11. **`reservation_status`**: Status final da reserva.
    - Valores: `'Check-Out'`, `'Canceled'`, `'No-Show'`

12. **`reservation_status_date`**: Data de atualização do último status, representada em formato de data.
    - Exemplo de valores: `'2015-07-01'`, `'2015-07-02'`, `'2015-07-03'`, `'2015-05-06'`, `'2015-04-22'`

## Dados Sensíveis

As colunas abaixo contêm informações pessoais (nomes e contatos) que são frequentemente pseudonimizadas para manter a privacidade do cliente:

1. **`name`**: Nome do hóspede.
   - Valores: `'Ernest Barnes'`, `'Andrea Baker'`, `'Rebecca Parker'`, ...

2. **`email`**: Endereço de e-mail do hóspede.
   - Valores: `'Ernest.Barnes31@outlook.com'`, `'Andrea_Baker94@aol.com'`, ...

3. **`phone-number`**: Número de telefone do hóspede.
   - Valores: `'669-792-1661'`, `'858-637-6955'`, `'652-885-2745'`, ...

4. **`credit_card`**: Número do cartão de crédito mascarado.
   - Valores: `'************4322'`, `'************9157'`, `'************3734'`, ...
