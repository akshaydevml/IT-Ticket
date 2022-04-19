from imports import *
from model import *

st.header('Daimler Auto-classification of IT tickets')
st.markdown('(created by the DTS-KREATE team)')

choice = st.sidebar.selectbox('Select Mode', ('Single', 'Excel File'))

if choice == 'Single':
    input_ticket = st.text_input(label='IT ticket to be Classified')
    if input_ticket:
        prediction = top_4_model.predict([input_ticket])
        if prediction:
            st.info(f"Predicted Class for the ticket: {pd.Series(prediction)[0]}")

elif choice == 'Excel File':
    tickets = st.file_uploader(label='Upload excel file containing tickets')
    if tickets:
        df_tickets = pd.read_excel(tickets)
        df_tickets = df_tickets.fillna(" ")

        predictions = []
        confidences = []

        tickets_list = df_tickets['Short Description'].tolist()
        ticket_ids = df_tickets['Ticket ID']

        for ticket in tickets_list:
            prediction = top_4_model.predict([ticket])
            confidence = np.max(top_4_model.predict_proba([ticket]))
            if confidence >= 0.86:
                predictions.append(prediction)
                confidences.append(confidence)
            elif confidence <= 0.86:
                prediction = top_40_model.predict([ticket])
                predictions.append(prediction)
                confidence = np.max(top_40_model.predict_proba([ticket]))
                confidences.append(confidence)

        predictions_df = pd.DataFrame({'Solved by Prediction': predictions})
        confidence_df = pd.DataFrame({'Confidence': confidences})
        ticket_id_df = pd.DataFrame({'Ticket ID' : ticket_ids})
        final_df = pd.concat([ticket_id_df['Ticket ID'], df_tickets['Short Description'], predictions_df['Solved by Prediction'], confidence_df['Confidence']], axis=1)

        st.write(final_df)
        final_df_2 = final_df.to_csv()
        st.download_button('Download Predictions', data=final_df_2)
